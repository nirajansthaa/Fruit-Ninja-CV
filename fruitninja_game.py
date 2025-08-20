# filename: fruit_ninja_mediapipe.py
import cv2, mediapipe as mp, numpy as np
import time, math, random, glob, os
from collections import deque

# -----------------------------
# Config
# -----------------------------
WIN_W, WIN_H = 1280, 720
SPAWN_EVERY_SEC = (1.5, 3.0)
GAME_DURATION_SEC   = 60            # 1 minute round
SPAWN_RAMP_MIN      = 0.45          # spawn interval shrinks to 45% by end
MULTI_SPAWN_THRESH  = [15, 30]     
MAX_ONSCREEN_FRUITS = 8        
RESTART_KEY         = (ord('r'), ord('R'))
QUIT_KEY = (27,)
BOMB_CHANCE = 0.15
FRUIT_MIN_SPEED = (-900, -1700)    # vy start (upwards are negative)
FRUIT_SIDE_SPEED = (-320, 320)    # vx range
GRAVITY = 2200
SLASH_VISUAL_COLOR     = (0, 0, 255)  
SLASH_VISUAL_THICKNESS = 8            
SLASH_GLOW_THICKNESS   = 14           
SLASH_GLOW_COLOR       = (0, 0, 120) 
ROT_SPEED = (2.9, 5.5)            # spin rad/s
FLASHBANG_DURATION   = 0.50   # seconds
FLASHBANG_ALPHA_MAX  = 220    # 0..255 (brightness of the white flash)
SHAKE_DURATION       = 0.35   # seconds
SHAKE_MAG_PX         = 18    
SLICE_SPEED_PX_PER_SEC = 1400
SLASH_THICKNESS = 140
SLASH_FRESHNESS_SEC = 0.12   # last fingertip sample must be newer than this
CLEAR_TRAIL_AFTER_HIT = True # clear trail once a cut is applied
CAM_INDEX = 2

# -----------------------------
def now(): return time.time()
def clamp(v, lo, hi): return max(lo, min(hi, v))

def ensure_rgba(img):
    if img is None: return None
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.full_like(b, 255)
        return cv2.merge((b,g,r,a))
    if img.shape[2] == 4:
        return img
    raise ValueError("Unsupported channels in fruit image")

def load_rgba_images(path, max_side=160):
    import glob, os
    files = []
    for ext in ("*.png", "*.webp"):
        files += glob.glob(os.path.join(path, ext))
    imgs = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im is None: continue
        if im.shape[2] == 3:
            b,g,r = cv2.split(im); a = np.full_like(b, 255); im = cv2.merge((b,g,r,a))
        # scale down big assets
        h,w = im.shape[:2]; s = max(h,w)
        if s > max_side:
            if w >= h: im = cv2.resize(im, (max_side, int(h*max_side/w)), interpolation=cv2.INTER_AREA)
            else:      im = cv2.resize(im, (int(w*max_side/h), max_side), interpolation=cv2.INTER_AREA)
        imgs.append(im)
    if not imgs:
        raise FileNotFoundError(f"No images in {path}")
    return imgs


def overlay_rgba(bg_bgr, fg_rgba, x, y):
    bh, bw = bg_bgr.shape[:2]
    fh, fw = fg_rgba.shape[:2]
    if x >= bw or y >= bh or x+fw <= 0 or y+fh <= 0:
        return
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(bw, x+fw); y1 = min(bh, y+fh)
    fx0 = x0 - x; fy0 = y0 - y
    fx1 = fx0 + (x1 - x0); fy1 = fy0 + (y1 - y0)

    roi = bg_bgr[y0:y1, x0:x1]
    fg = fg_rgba[fy0:fy1, fx0:fx1, :3].astype(np.float32)
    a  = (fg_rgba[fy0:fy1, fx0:fx1, 3:4].astype(np.float32) / 255.0)
    inv= 1.0 - a
    roi[:] = (a*fg + inv*roi.astype(np.float32)).astype(np.uint8)

def y_axis_spin_warp(rgba, theta):
    h, w = rgba.shape[:2]
    scale_x = 0.35 + 0.65*abs(math.cos(theta))
    new_w = max(8, int(w*scale_x))
    warped = cv2.resize(rgba, (new_w, h), interpolation=cv2.INTER_LINEAR)
    if math.cos(theta) < 0:
        warped = cv2.flip(warped, 1)
    return warped

def split_image_by_line(rgba, angle_rad):
    h, w = rgba.shape[:2]
    cx, cy = w/2.0, h/2.0
    vx, vy = math.cos(angle_rad), math.sin(angle_rad)
    nx, ny = -vy, vx  # line normal

    xs = np.arange(w) - cx
    ys = np.arange(h) - cy
    X, Y = np.meshgrid(xs, ys)
    signed = X*nx + Y*ny

    left_mask = (signed <= 0).astype(np.uint8)
    right_mask = (signed > 0).astype(np.uint8)

    left = np.zeros_like(rgba)
    right = np.zeros_like(rgba)
    for c in range(4):
        left[..., c]  = rgba[..., c] * left_mask
        right[..., c] = rgba[..., c] * right_mask
    return left, right, (nx, ny)

def add_juice_tint(rgba, tint_bgr=(40,120,240), alpha_strength=50):
    # overlay semi-transparent tint (quick "juice" boost)
    tint = np.zeros_like(rgba)
    tint[..., :3] = np.array(tint_bgr, dtype=np.uint8)
    tint[..., 3] = alpha_strength
    out = rgba.copy()
    overlay_rgba(out[..., :3], tint, 0, 0)
    return out

def dist_point_to_segment(px, py, ax, ay, bx, by):
    apx, apy = px-ax, py-ay
    abx, aby = bx-ax, by-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0:
        return math.hypot(apx, apy)
    t = clamp((apx*abx + apy*aby) / ab2, 0.0, 1.0)
    cx, cy = ax + t*abx, ay + t*aby
    return math.hypot(px-cx, py-cy)

# -----------------------------
# Game objects
# -----------------------------
class Particle:
    def __init__(self, pos, vel, life=0.3):
        self.x, self.y = pos
        self.vx, self.vy = vel
        self.t0 = now()
        self.life = life
    def alive(self): return (now() - self.t0) < self.life
    def update(self, dt):
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.vy += GRAVITY*0.3*dt
    def draw(self, frame):
        age = now() - self.t0
        a = 1.0 - (age/self.life)
        r = int(3 + 8*(a))
        if r > 0:
            cv2.circle(frame, (int(self.x), int(self.y)), r, (40,120,240), -1, lineType=cv2.LINE_AA)

class BoomParticle:
    def __init__(self, pos, vel, life=0.45, color=(30,30,30)):
        self.x, self.y = pos
        self.vx, self.vy = vel
        self.t0 = now(); self.life = life
        self.color = color
    def alive(self): return (now()-self.t0) < self.life
    def update(self, dt):
        self.x += self.vx*dt; self.y += self.vy*dt
        # slight drag
        self.vx *= (1 - 2.0*dt); self.vy *= (1 - 2.0*dt)
    def draw(self, frame):
        a = 1.0 - (now()-self.t0)/self.life
        r = max(1, int(6*a))
        cv2.circle(frame, (int(self.x), int(self.y)), r, self.color, -1, lineType=cv2.LINE_AA)

class Bomb:
    def __init__(self, rgba_choices):
        self.base_rgba = random.choice(rgba_choices)
        self.x = random.randint(150, WIN_W-150)
        self.y = WIN_H + 40
        self.vx = random.uniform(*FRUIT_SIDE_SPEED)
        self.vy = random.uniform(*FRUIT_MIN_SPEED)
        self.theta = random.uniform(0, math.tau)
        self.rot_speed = random.uniform(*ROT_SPEED) * random.choice([-1,1])
        self.dead = False
        self.triggered = False
        h,w = self.base_rgba.shape[:2]
        self.radius = max(h,w)//2
    def update(self, dt):
        if self.triggered: return
        self.x += self.vx*dt; self.y += self.vy*dt
        self.vy += GRAVITY*dt; self.theta += self.rot_speed*dt
        if self.y - 80 > WIN_H: self.dead = True
    def draw(self, frame):
        if self.triggered: return
        spin = y_axis_spin_warp(self.base_rgba, self.theta)
        overlay_rgba(frame, spin, int(self.x - spin.shape[1]/2), int(self.y - spin.shape[0]/2))
    def bbox_and_radius(self):
        return (int(self.x), int(self.y), self.radius)

class FruitHalf:
    def __init__(self, rgba, pos, vel, spin=0.0, spin_speed=0.0):
        self.rgba = rgba
        self.x, self.y = pos
        self.vx, self.vy = vel
        self.spin = spin
        self.spin_speed = spin_speed
        self.dead = False
    def update(self, dt):
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.vy += GRAVITY*dt
        self.spin += self.spin_speed*dt
        if self.y - 60 > WIN_H: self.dead = True
    def draw(self, frame):
        M = cv2.getRotationMatrix2D((self.rgba.shape[1]/2, self.rgba.shape[0]/2),
                                    math.degrees(self.spin), 1.0)
        rotated = cv2.warpAffine(self.rgba, M, (self.rgba.shape[1], self.rgba.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        overlay_rgba(frame, rotated, int(self.x - rotated.shape[1]/2), int(self.y - rotated.shape[0]/2))

class Fruit:
    def __init__(self, rgba_choices):
        self.base_rgba = random.choice(rgba_choices)
        self.x = random.randint(150, WIN_W-150)
        self.y = WIN_H + 40
        self.vx = random.uniform(*FRUIT_SIDE_SPEED)
        self.vy = random.uniform(*FRUIT_MIN_SPEED)
        self.theta = random.uniform(0, math.tau)
        self.rot_speed = random.uniform(*ROT_SPEED) * random.choice([-1,1])
        self.dead = False
        self.sliced = False

        # rough “juice” tint from average color (inverse-ish)
        avg = np.mean(self.base_rgba[..., :3].reshape(-1,3), axis=0)
        self.juice_bgr = tuple(int(255 - c) for c in avg[::-1])

        # approximate radius from image size
        h,w = self.base_rgba.shape[:2]
        self.radius = max(h,w)//2

    def update(self, dt):
        if self.sliced: return
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.vy += GRAVITY*dt
        self.theta += self.rot_speed*dt
        if self.y - 80 > WIN_H: self.dead = True

    def draw(self, frame):
        if self.sliced: return
        spin = y_axis_spin_warp(self.base_rgba, self.theta)
        overlay_rgba(frame, spin, int(self.x - spin.shape[1]/2), int(self.y - spin.shape[0]/2))

    def bbox_and_radius(self):
        return (int(self.x), int(self.y), self.radius)

    def slice(self, slash_angle, vx_hint, vy_hint):
        self.sliced = True
        rgba = add_juice_tint(self.base_rgba, tint_bgr=self.juice_bgr, alpha_strength=40)
        left, right, normal = split_image_by_line(rgba, slash_angle)
        nx, ny = normal
        kick = 380
        h1 = FruitHalf(left, (self.x, self.y), (self.vx + vx_hint - nx*kick, self.vy + vy_hint - ny*kick),
                       spin=0.0, spin_speed=random.uniform(-6,6))
        h2 = FruitHalf(right,(self.x, self.y), (self.vx + vx_hint + nx*kick, self.vy + vy_hint + ny*kick),
                       spin=0.0, spin_speed=random.uniform(-6,6))
        parts = []
        for _ in range(16):
            spd = random.uniform(120, 360)
            ang = slash_angle + (random.uniform(-0.7, 0.7) + (math.pi/2))*random.choice([-1,1])
            parts.append(Particle((self.x, self.y), (math.cos(ang)*spd, math.sin(ang)*spd),
                                  life=random.uniform(0.25,0.45)))
        return h1, h2, parts

# -----------------------------
# Slash detection
# -----------------------------
class SlashTracker:
    def __init__(self, history_sec=0.25):
        from collections import deque
        self.hist = deque(maxlen=64)  # (x,y,t)
        self.history_sec = history_sec
        self.last_seen = 0.0

    def _trim(self, t):
        while self.hist and (t - self.hist[0][2]) > self.history_sec:
            self.hist.popleft()

    def add(self, x, y, t):
        self.hist.append((x,y,t))
        while self.hist and (t - self.hist[0][2]) > self.history_sec:
            self.hist.popleft()

    def clear_if_stale(self, t, stale_after=0.12):
        # If we haven't seen a hand recently, forget the trail
        if (t - self.last_seen) > stale_after:
            self.hist.clear()

    def current_speed_and_segment(self, t , freshness=0.12):
        if len(self.hist) < 2: return 0.0, None
        p1 = self.hist[-1]
        if (t - p1[2]) > freshness:
            return 0.0, None
        
        for i in range(len(self.hist)-2, -1, -1):
            if (p1[2] - self.hist[i][2]) >= 0.03:
                p0 = self.hist[i]
                dt = max(1e-3, p1[2]-p0[2])
                speed = math.hypot(p1[0]-p0[0], p1[1]-p0[1]) / dt
                return speed, (p0, p1)
        return 0.0, None
    
    def draw_trail(self, frame):
        pts = [(int(x), int(y)) for (x, y, _) in self.hist]
        if len(pts) < 2:
            return

        # Optional: soft red "glow" under the main line (draw thicker + darker first)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], SLASH_GLOW_COLOR, SLASH_GLOW_THICKNESS, lineType=cv2.LINE_AA)

        # Main bright red trail on top
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], SLASH_VISUAL_COLOR, SLASH_VISUAL_THICKNESS, lineType=cv2.LINE_AA)


# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands

def get_index_tip_xy(game_canvas_bgr, hands):
    """Run detection on the same (resized) canvas to keep coordinates aligned. No mirroring."""
    rgb = cv2.cvtColor(game_canvas_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks: return None
    h, w = game_canvas_bgr.shape[:2]
    pt = res.multi_hand_landmarks[0].landmark[8]  # index tip
    return (int(pt.x * w), int(pt.y * h))

# -----------------------------
def main():
    # Load fruit images (RGBA)
    fruit_imgs = load_rgba_images("assets/fruits", max_side=170)
    bomb_imgs  = load_rgba_images("assets/bomb",  max_side=170) 

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fruits = []
    halves = []
    particles = []
    bombs = []
    flashbang_timer = 0.0   
    shake_timer  = 0.0   
    flash_timer = 0.0
    score = 0
    game_over = False
    game_start_time = now()
    time_left = GAME_DURATION_SEC
    next_spawn_at = now() + random.uniform(*SPAWN_EVERY_SEC)
    slash = SlashTracker()
    last_time = now()

    def reset_game():
        nonlocal fruits, bombs, halves, particles
        nonlocal score, flash_timer, game_over, game_start_time, time_left, next_spawn_at
        fruits.clear(); bombs.clear(); halves.clear(); particles.clear()
        score = 0
        flash_timer = 0.0
        game_over = False
        game_start_time = now()
        time_left = GAME_DURATION_SEC
        next_spawn_at = now() + 0.6

    with mp_hands.Hands(model_complexity=0, max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, cam = cap.read()
            if not ok:
                cam = np.zeros((480, 640, 3), dtype=np.uint8)
            
            cam = cv2.flip(cam, 1)  

            # Resize camera to game canvas size and USE IT AS BACKGROUND
            frame = cv2.resize(cam, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

            # time step
            t = now()
            dt = min(0.033, t - last_time)
            last_time = t

            elapsed   = t - game_start_time
            time_left = max(0.0, GAME_DURATION_SEC - elapsed)
            if (time_left <= 0.0) and not game_over:
                game_over = True 

            # spawn
            if (not game_over) and (t >= next_spawn_at) and (len(fruits) + len(bombs) < MAX_ONSCREEN_FRUITS):
                ramp = SPAWN_RAMP_MIN + (1.0 - SPAWN_RAMP_MIN) * (time_left / GAME_DURATION_SEC)

                spawns = 1 + (1 if elapsed >= MULTI_SPAWN_THRESH[0] else 0) + (1 if elapsed >= MULTI_SPAWN_THRESH[1] else 0)

                for _ in range(spawns):
                    if random.random() < BOMB_CHANCE:
                        bombs.append(Bomb(bomb_imgs))
                    else:
                        fruits.append(Fruit(fruit_imgs))

                next_spawn_at = t + random.uniform(SPAWN_EVERY_SEC[0]*ramp, SPAWN_EVERY_SEC[1]*ramp)

            # fingertip on SAME canvas
            tip_xy = get_index_tip_xy(frame, hands)
            if tip_xy:
                slash.add(tip_xy[0], tip_xy[1], t)
            else:
                slash.clear_if_stale(t, SLASH_FRESHNESS_SEC)

            speed, seg = slash.current_speed_and_segment(t, SLASH_FRESHNESS_SEC)

            # update/draw fruits over webcam background
            for fr in fruits:
                fr.update(dt)
                fr.draw(frame)

            for b in bombs:
                b.update(dt)
                b.draw(frame)

            # slice
            if (not game_over) and seg and speed > SLICE_SPEED_PX_PER_SEC:
                (x0,y0,_), (x1,y1,_) = seg
                angle = math.atan2((y1-y0), (x1-x0))
                vx_hint = 0.0; vy_hint = 0.0
            
                did_hit = False

                # First: check bombs — if hit, explode & penalize
                for b in bombs:
                    if b.dead or b.triggered: continue
                    cx, cy, r = b.bbox_and_radius()
                    d = dist_point_to_segment(cx, cy, x0, y0, x1, y1)
                    if d <= r + SLASH_THICKNESS:
                        b.triggered = True
                        score = max(0, score - 3)
                        did_hit = True
                        flashbang_timer = FLASHBANG_DURATION
                        shake_timer     = SHAKE_DURATION
                            

                        # big radial burst
                        for _ in range(36):
                            spd = random.uniform(300, 900)
                            ang = random.uniform(0, math.tau)
                            particles.append(BoomParticle((b.x, b.y), (math.cos(ang)*spd, math.sin(ang)*spd),
                                                        life=random.uniform(0.25, 0.5), color=(20,20,20)))
                        break  # one bomb per slash (optional)

                # Then: slice fruits
                for fr in fruits:
                    if fr.dead or fr.sliced: continue
                    cx, cy, r = fr.bbox_and_radius()
                    d = dist_point_to_segment(cx, cy, x0, y0, x1, y1)
                    if d <= r + SLASH_THICKNESS:
                        h1, h2, pts = fr.slice(angle, vx_hint, vy_hint)
                        halves.extend([h1, h2]); particles.extend(pts); score += 1
                
                if did_hit and CLEAR_TRAIL_AFTER_HIT:
                    slash.hist.clear()

            # remove sliced or dead fruits
            fruits = [f for f in fruits if not (f.dead or f.sliced)]
            bombs  = [b for b in bombs  if not (b.dead or b.triggered)]


            # halves & particles
            kept_halves = []
            for h in halves:
                h.update(dt); h.draw(frame)
                if not h.dead: kept_halves.append(h)
            halves = kept_halves

            kept_parts = []
            for p in particles:
                if p.alive():
                    p.update(dt); p.draw(frame)
                    kept_parts.append(p)
            particles = kept_parts

            # trail + HUD (over webcam)
            slash.draw_trail(frame)
            cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,180), 2, cv2.LINE_AA)

            mm = int(time_left // 60)
            ss = int(time_left % 60)
            timer_text = f"{mm}:{ss:02d}"
            (tw, th), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(frame, timer_text, (WIN_W - tw - 20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

            if game_over:
                # dim
                overlay = np.zeros((WIN_H, WIN_W, 4), dtype=np.uint8)
                overlay[..., :3] = (0, 0, 0)
                overlay[..., 3]   = 140
                overlay_rgba(frame, overlay, 0, 0)

                # text
                cv2.putText(frame, "TIME UP!", (WIN_W//2 - 160, WIN_H//2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 4, cv2.LINE_AA)
                cv2.putText(frame, "Press R to Restart, Esc to Quit",
                            (WIN_W//2 - 330, WIN_H//2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                
            if shake_timer > 0.0:
                k = shake_timer / SHAKE_DURATION             # 1 -> 0 over time
                strength = int(SHAKE_MAG_PX * k)             # ease-out
                dx = random.randint(-strength, strength)
                dy = random.randint(-strength, strength)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(frame, M, (WIN_W, WIN_H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)  # nicer edges
                shake_timer = max(0.0, shake_timer - dt)

# ---- Flashbang (white overlay with fading alpha) ----
            if flashbang_timer > 0.0:
                k = flashbang_timer / FLASHBANG_DURATION     # 1 -> 0
                alpha = int(FLASHBANG_ALPHA_MAX * (k * k))   # quadratic ease-out
                overlay = np.zeros((WIN_H, WIN_W, 4), dtype=np.uint8)
                overlay[..., :3] = (255, 255, 255)           # white flash
                overlay[..., 3]  = alpha
                overlay_rgba(frame, overlay, 0, 0)
                flashbang_timer = max(0.0, flashbang_timer - dt)
            cv2.imshow("Live Fruit Ninja", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEY:
                break
            if key in RESTART_KEY:
                reset_game()
                continue


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
