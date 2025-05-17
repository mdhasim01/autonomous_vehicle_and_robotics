import pygame
import sys
import math
import random
import time
import numpy as np
import os
from data_collector import DataCollector
try:
    from ml_models import BehaviorCloningModel, HazardDetectionModel
    ML_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Machine Learning features will be disabled.")
    ML_AVAILABLE = False

# --- Constants ---
# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
# World dimensions (larger than screen for camera movement)
WORLD_WIDTH, WORLD_HEIGHT = 4000, 3000

# Colors (Enhanced Palette)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ROAD_GRAY = (80, 80, 85)      # Darker road
ROAD_BORDER = (50, 50, 55)    # Even darker border
LANE_YELLOW = (255, 220, 0)   # Lane markings
GRASS_GREEN = (50, 150, 50)   # Base grass
DARK_GRASS = (30, 110, 30)    # Darker grass patches
BUILDING_BLUE = (70, 100, 180) # Muted building blue
BUILDING_ACCENT = (100, 130, 210) # Lighter accent
BUILDING_WINDOW = (170, 200, 255) # Window color
TREE_TRUNK = (100, 60, 20)
TREE_GREEN_1 = (0, 140, 0)
TREE_GREEN_2 = (0, 110, 0)
SIGN_POST = (90, 90, 90)
SIGN_FACE = (230, 230, 230)
PLOTHOL_COLOR = (40, 40, 40) # Dark, irregular
SPEED_BREAKER_COLOR = (150, 150, 0) # Yellowish bumps
WARNING_RED = (255, 40, 40, 220)   # Warning messages background (semi-transparent)
WARNING_YELLOW = (255, 220, 0) # Warning messages text
WARNING_BORDER = (200, 0, 0)  # Warning message border

VEHICLE_RED = (220, 30, 30)
VEHICLE_DARK_RED = (150, 0, 0)
VEHICLE_WINDOW = (150, 180, 240)
VEHICLE_LIGHT = (255, 255, 150) # Headlights

SENSOR_LIDAR_COLOR = (0, 255, 255) # Cyan
SENSOR_CAMERA_COLOR = (255, 255, 0, 70) # Transparent Yellow
POPUP_BG = (30, 30, 70, 200) # Semi-transparent dark blue
POPUP_TEXT = (200, 200, 255)

# Vehicle properties
VEHICLE_WIDTH = 25
VEHICLE_HEIGHT = 50
VEHICLE_ACCEL = 0.05        # Auto acceleration
VEHICLE_BRAKE_ACCEL = 0.15  # Auto braking
VEHICLE_DECEL = 0.02        # Natural friction/drag
VEHICLE_MAX_SPEED = 3.5     # Auto max speed
VEHICLE_TURN_RATE = 2.5     # Degrees per frame for path following (scaled by speed)
VEHICLE_EVADE_TURN_RATE = 4.0 # Faster turning during evasion
VEHICLE_LANE_CHANGE_SPEED = 1.8 # Lateral speed during evasion
LANE_WIDTH = 40             # Approximate width for lane change logic

# Manual Control Properties
MANUAL_ACCEL_MULT = 1.8     # Manual acceleration multiplier
MANUAL_BRAKE_MULT = 1.5     # Manual braking multiplier
MANUAL_TURN_MULT = 1.5      # Manual turning multiplier
MANUAL_MAX_SPEED_MULT = 1.1 # Manual max speed multiplier

# Hazard properties
PLOTHOL_DETECT_RANGE = 120
SPEED_BREAKER_DETECT_RANGE = 150
SPEED_BREAKER_SLOW_SPEED = 1.0 # Target speed for auto mode over breaker
MANUAL_SPEED_BREAKER_THRESHOLD = 1.8 # Max speed for manual over speed breaker
MANUAL_OFF_TRACK_THRESHOLD = LANE_WIDTH * 1.8 # How far off path center is considered "off track"

# Sensor properties
LIDAR_RANGE = 180
LIDAR_ANGLE_SPREAD = 140
LIDAR_NUM_RAYS = 25
CAMERA_FOV_ANGLE = 80
CAMERA_RANGE = 200

# Camera properties
CAMERA_LAG = 0.08 # Smoother camera

# Popup & Warning properties
POPUP_DURATION = 2.0 # seconds
POPUP_FADE_TIME = 0.5 # seconds (part of duration)
WARNING_DURATION = 2.5 # seconds for technical warnings
WARNING_COOLDOWN = 1.0 # Minimum time between warnings

# Control Modes
AUTO = "AUTOMATIC"
MANUAL = "MANUAL"

# Control Keys
KEY_ACCEL = pygame.K_UP
KEY_BRAKE = pygame.K_DOWN
KEY_LEFT = pygame.K_LEFT
KEY_RIGHT = pygame.K_RIGHT
KEY_MODE_SWITCH = pygame.K_m

# --- Pygame Initialization ---
try:
    pygame.init()
    pygame.font.init()
except (pygame.error, pygame.font.error) as e:
    print(f"Fatal Error: Failed to initialize Pygame or Font module: {e}")
    sys.exit(1)

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Interactive Autonomous Simulation Prototype")
# Create a separate surface for alpha blending (sensors, popups, warnings)
alpha_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

# Fonts
try:
    default_font = pygame.font.SysFont("Consolas", 18)
    popup_font = pygame.font.SysFont("Consolas", 20, bold=True)
    warning_font = pygame.font.SysFont("Consolas", 36, bold=True)
except Exception:
    print("Warning: Consolas font not found, using default pygame font.")
    default_font = pygame.font.Font(None, 20)
    popup_font = pygame.font.Font(None, 22)
    warning_font = pygame.font.Font(None, 40)

# --- Helper Functions ---
def world_to_screen(pos, camera_offset):
    """Converts world coordinates to screen coordinates."""
    return (int(pos[0] - camera_offset[0]), int(pos[1] - camera_offset[1]))

def screen_to_world(pos, camera_offset):
    """Converts screen coordinates to world coordinates."""
    return (pos[0] + camera_offset[0], pos[1] + camera_offset[1])

def rotate_point(center, point, angle_degrees):
    """Rotates a point around a center by a given angle in degrees."""
    angle_rad = math.radians(angle_degrees)
    cx, cy = center
    px, py = point
    x_new = cx + (px - cx) * math.cos(angle_rad) - (py - cy) * math.sin(angle_rad)
    y_new = cy + (px - cx) * math.sin(angle_rad) + (py - cy) * math.cos(angle_rad)
    return x_new, y_new

def lerp(start, end, factor):
    """Linear interpolation."""
    return start + (end - start) * factor

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle_diff(angle1, angle2):
    """ Calculate the shortest difference between two angles (-180 to 180) """
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff

def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10, gap_length=5):
    """Draws a dashed line between two points."""
    dist = distance(start_pos, end_pos)
    if dist == 0 or dash_length <= 0: return
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    segment_length = dash_length + gap_length
    num_segments = int(dist / segment_length)

    for i in range(num_segments):
        t0 = i * segment_length / dist
        t1 = (i * segment_length + dash_length) / dist
        if t1 > 1.0: t1 = 1.0 # Clamp last segment
        p0 = (start_pos[0] + dx * t0, start_pos[1] + dy * t0)
        p1 = (start_pos[0] + dx * t1, start_pos[1] + dy * t1)
        try:
            pygame.draw.line(surf, color, p0, p1, width)
        except TypeError: # Handle potential issues with float coords in older pygame
             pygame.draw.line(surf, color, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), width)
    # Draw last partial dash if needed and gap exists
    if num_segments * segment_length < dist - gap_length:
        t0 = num_segments * segment_length / dist
        t1 = min(1.0, (num_segments * segment_length + dash_length) / dist)
        p0 = (start_pos[0] + dx * t0, start_pos[1] + dy * t0)
        p1 = (start_pos[0] + dx * t1, start_pos[1] + dy * t1)
        try:
            pygame.draw.line(surf, color, p0, p1, width)
        except TypeError:
             pygame.draw.line(surf, color, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), width)

def point_segment_distance(px, py, x1, y1, x2, y2):
    """Calculates the shortest distance from a point (px, py) to a line segment ((x1,y1), (x2,y2))."""
    line_mag_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_mag_sq < 0.000001: # Segment is a point
        return distance((px, py), (x1, y1))

    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_mag_sq
    u = max(0, min(1, u)) # Clamp u to the segment

    closest_x = x1 + u * (x2 - x1)
    closest_y = y1 + u * (y2 - y1)

    return distance((px, py), (closest_x, closest_y))


# --- Path Definition ---
# More complex path with sharper turns and varied segment lengths
path_points = [
    (WORLD_WIDTH * 0.1, WORLD_HEIGHT * 0.15),
    (WORLD_WIDTH * 0.4, WORLD_HEIGHT * 0.15), # Straight
    (WORLD_WIDTH * 0.5, WORLD_HEIGHT * 0.2),  # Gentle curve
    (WORLD_WIDTH * 0.55, WORLD_HEIGHT * 0.35), # Sharper curve
    (WORLD_WIDTH * 0.55, WORLD_HEIGHT * 0.5), # Straight
    (WORLD_WIDTH * 0.8, WORLD_HEIGHT * 0.5),  # Long straight
    (WORLD_WIDTH * 0.9, WORLD_HEIGHT * 0.55), # Curve
    (WORLD_WIDTH * 0.9, WORLD_HEIGHT * 0.7),  # Straight
    (WORLD_WIDTH * 0.8, WORLD_HEIGHT * 0.85), # Sharp curve
    (WORLD_WIDTH * 0.6, WORLD_HEIGHT * 0.88), # Short straight
    (WORLD_WIDTH * 0.4, WORLD_HEIGHT * 0.8),  # Curve
    (WORLD_WIDTH * 0.25, WORLD_HEIGHT * 0.75),# Curve
    (WORLD_WIDTH * 0.1, WORLD_HEIGHT * 0.6),  # Curve
    (WORLD_WIDTH * 0.05, WORLD_HEIGHT * 0.4), # Sharp curve back
]

# --- Environment Generation ---
def create_environment(width, height):
    """Creates the static world surface with roads, obstacles, and high-density hazards."""
    world_surface = pygame.Surface((width, height))
    world_surface.fill(DARK_GRASS)

    # Draw grass patches for texture
    for _ in range(250): # More patches
        patch_x = random.randint(0, width)
        patch_y = random.randint(0, height)
        patch_w = random.randint(40, 180)
        patch_h = random.randint(40, 180)
        pygame.draw.ellipse(world_surface, GRASS_GREEN, (patch_x, patch_y, patch_w, patch_h))

    obstacles = [] # Store static obstacle rects (buildings, trees, etc.)
    hazards = [] # Store hazard info: (type, rect, center)

    # Draw the road network
    road_total_width = LANE_WIDTH * 2 + 10 # Two lanes + shoulder/margin
    center_line_width = 4
    # Draw thick lines for border, then road, then lanes
    pygame.draw.lines(world_surface, ROAD_BORDER, True, path_points, road_total_width + 15) # Wider border
    pygame.draw.lines(world_surface, ROAD_GRAY, True, path_points, road_total_width)

    # Draw dashed center line
    for i in range(len(path_points)):
        p1 = path_points[i]
        p2 = path_points[(i + 1) % len(path_points)]
        draw_dashed_line(world_surface, LANE_YELLOW, p1, p2, center_line_width, 15, 10)

    # Add static obstacles (optional, for visual clutter)
    obstacle_definitions = [
        # Type, Rect (x, y, w, h), Color1, Color2, Color3 (window)
        ("building", pygame.Rect(WORLD_WIDTH*0.2, WORLD_HEIGHT*0.05, 150, 100), BUILDING_BLUE, BUILDING_ACCENT, BUILDING_WINDOW),
        ("building", pygame.Rect(WORLD_WIDTH*0.6, WORLD_HEIGHT*0.05, 120, 180), BUILDING_BLUE, BUILDING_ACCENT, BUILDING_WINDOW),
        ("building", pygame.Rect(WORLD_WIDTH*0.7, WORLD_HEIGHT*0.6, 200, 150), BUILDING_BLUE, BUILDING_ACCENT, BUILDING_WINDOW),
        ("building", pygame.Rect(WORLD_WIDTH*0.15, WORLD_HEIGHT*0.85, 180, 130), BUILDING_BLUE, BUILDING_ACCENT, BUILDING_WINDOW),
        ("tree", pygame.Rect(WORLD_WIDTH*0.35, WORLD_HEIGHT*0.5, 40, 40), TREE_TRUNK, TREE_GREEN_1, TREE_GREEN_2), # Rect is canopy size
        ("tree", pygame.Rect(WORLD_WIDTH*0.55, WORLD_HEIGHT*0.9, 50, 50), TREE_TRUNK, TREE_GREEN_1, TREE_GREEN_2),
        ("tree", pygame.Rect(WORLD_WIDTH*0.95, WORLD_HEIGHT*0.2, 35, 35), TREE_TRUNK, TREE_GREEN_1, TREE_GREEN_2),
        ("tree", pygame.Rect(WORLD_WIDTH*0.05, WORLD_HEIGHT*0.5, 45, 45), TREE_TRUNK, TREE_GREEN_1, TREE_GREEN_2),
    ]
    for definition in obstacle_definitions:
        obj_type, rect, color1, color2, color3 = definition
        obstacles.append(rect)
        if obj_type == "building":
            pygame.draw.rect(world_surface, color1, rect)
            pygame.draw.rect(world_surface, color2, rect.inflate(-10, -10))
            win_w, win_h = 15, 25
            for row in range(int(rect.height / (win_h + 15))):
                for col in range(int(rect.width / (win_w + 15))):
                    win_x = rect.left + 10 + col * (win_w + 15)
                    win_y = rect.top + 10 + row * (win_h + 15)
                    if win_x + win_w < rect.right - 10 and win_y + win_h < rect.bottom - 10:
                         pygame.draw.rect(world_surface, color3, (win_x, win_y, win_w, win_h))
        elif obj_type == "tree":
            trunk_h = rect.height * 0.4
            trunk_w = rect.width * 0.2
            trunk_rect = pygame.Rect(rect.centerx - trunk_w // 2, rect.bottom - trunk_h, trunk_w, trunk_h)
            pygame.draw.rect(world_surface, color1, trunk_rect)
            pygame.draw.circle(world_surface, color2, (rect.centerx, rect.centery - rect.height * 0.1), rect.width // 2)
            pygame.draw.circle(world_surface, color3, (rect.centerx - rect.width*0.2, rect.centery + rect.height*0.1), rect.width // 3)
            pygame.draw.circle(world_surface, color2, (rect.centerx + rect.width*0.15, rect.centery), rect.width // 2.5)

    # Add HIGH DENSITY Hazards (Plotholes, Speed Breakers) - Placed strategically
    hazard_definitions = [
        # Type, Center Position, Size (radius or width/height)

        # Section 1 (Start Straight) - Dense potholes
        ("plothole", (WORLD_WIDTH * 0.20, WORLD_HEIGHT * 0.15), 18),
        ("plothole", (WORLD_WIDTH * 0.28, WORLD_HEIGHT * 0.16), 22), # Offset
        ("plothole", (WORLD_WIDTH * 0.35, WORLD_HEIGHT * 0.14), 20), # Offset other way

        # Section 2 (Curves 1 & 2) - Breaker after turn, potholes in curve
        ("speedbreaker", (WORLD_WIDTH * 0.45, WORLD_HEIGHT * 0.17), (road_total_width, 15)), # Just entering curve
        ("plothole", (WORLD_WIDTH * 0.52, WORLD_HEIGHT * 0.25), 25), # Mid-curve
        ("plothole", (WORLD_WIDTH * 0.55, WORLD_HEIGHT * 0.32), 18), # Mid-curve
        ("speedbreaker", (WORLD_WIDTH * 0.55, WORLD_HEIGHT * 0.42), (road_total_width, 15)), # After curve

        # Section 3 (Long Straight) - Sequence of hazards
        ("plothole", (WORLD_WIDTH * 0.60, WORLD_HEIGHT * 0.51), 20),
        ("speedbreaker", (WORLD_WIDTH * 0.65, WORLD_HEIGHT * 0.50), (road_total_width, 15)),
        ("plothole", (WORLD_WIDTH * 0.70, WORLD_HEIGHT * 0.49), 24),
        ("plothole", (WORLD_WIDTH * 0.75, WORLD_HEIGHT * 0.51), 18),
        ("speedbreaker", (WORLD_WIDTH * 0.78, WORLD_HEIGHT * 0.50), (road_total_width, 15)),

        # Section 4 (Curve 3 & Straight) - Potholes in curve, breaker
        ("plothole", (WORLD_WIDTH * 0.88, WORLD_HEIGHT * 0.58), 22), # In curve
        ("plothole", (WORLD_WIDTH * 0.90, WORLD_HEIGHT * 0.65), 19), # In curve
        ("speedbreaker", (WORLD_WIDTH * 0.90, WORLD_HEIGHT * 0.75), (road_total_width, 15)), # After curve

        # Section 5 (Sharp Curve 4 & Straight) - Very challenging section
        ("plothole", (WORLD_WIDTH * 0.75, WORLD_HEIGHT * 0.88), 28), # Right after sharp turn
        ("speedbreaker", (WORLD_WIDTH * 0.65, WORLD_HEIGHT * 0.89), (road_total_width, 15)), # Close sequence
        ("plothole", (WORLD_WIDTH * 0.58, WORLD_HEIGHT * 0.87), 20), # Close sequence
        ("plothole", (WORLD_WIDTH * 0.50, WORLD_HEIGHT * 0.85), 25), # Close sequence

        # Section 6 (Curves 5 & 6)
        ("speedbreaker", (WORLD_WIDTH * 0.35, WORLD_HEIGHT * 0.78), (road_total_width, 15)), # Mid curve
        ("plothole", (WORLD_WIDTH * 0.28, WORLD_HEIGHT * 0.72), 21),
        ("plothole", (WORLD_WIDTH * 0.15, WORLD_HEIGHT * 0.65), 26), # Large one

        # Section 7 (Curve 7 & Final Curve)
        ("speedbreaker", (WORLD_WIDTH * 0.08, WORLD_HEIGHT * 0.50), (road_total_width, 15)),
        ("plothole", (WORLD_WIDTH * 0.06, WORLD_HEIGHT * 0.35), 23), # In final sharp curve
        ("plothole", (WORLD_WIDTH * 0.08, WORLD_HEIGHT * 0.25), 19), # In final sharp curve
        ("speedbreaker", (WORLD_WIDTH * 0.1, WORLD_HEIGHT * 0.18), (road_total_width, 15)), # Near start/finish
    ]

    for hz_type, hz_center, hz_size in hazard_definitions:
        if hz_type == "plothole":
            radius = hz_size
            hz_rect = pygame.Rect(hz_center[0] - radius, hz_center[1] - radius, radius * 2, radius * 2)
            # Draw irregular plothole shape
            num_points = random.randint(7, 11)
            points = []
            for i in range(num_points):
                angle = (i / num_points) * 2 * math.pi + random.uniform(-0.2, 0.2)
                r = radius * random.uniform(0.75, 1.15)
                px = hz_center[0] + r * math.cos(angle)
                py = hz_center[1] + r * math.sin(angle)
                points.append((px, py))
            try:
                pygame.draw.polygon(world_surface, PLOTHOL_COLOR, points)
                pygame.draw.polygon(world_surface, ROAD_BORDER, points, 1) # Subtle border
            except ValueError:
                print(f"Warning: Could not draw plothole polygon at {hz_center}") # Handle potential errors
            hazards.append(("plothole", hz_rect, hz_center))
        elif hz_type == "speedbreaker":
            width, height = hz_size
            # Position breaker perpendicular to path segment (approximate)
            # Find closest path segment and its angle
            min_d = float('inf')
            segment_angle = 0
            for i in range(len(path_points)):
                p1 = path_points[i]
                p2 = path_points[(i + 1) % len(path_points)]
                d = point_segment_distance(hz_center[0], hz_center[1], p1[0], p1[1], p2[0], p2[1])
                if d < min_d:
                    min_d = d
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    segment_angle = math.degrees(math.atan2(dy, dx))

            # Create points for a rotated rectangle
            angle_rad = math.radians(segment_angle)
            w, h = width, height
            points = [
                (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)
            ]
            rotated_points = [rotate_point((0,0), p, segment_angle) for p in points]
            world_points = [(p[0] + hz_center[0], p[1] + hz_center[1]) for p in rotated_points]

            # Calculate bounding box for hazard list
            min_x = min(p[0] for p in world_points)
            max_x = max(p[0] for p in world_points)
            min_y = min(p[1] for p in world_points)
            max_y = max(p[1] for p in world_points)
            hz_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

            # Draw the rotated rectangle (polygon)
            pygame.draw.polygon(world_surface, SPEED_BREAKER_COLOR, world_points)
            # Add stripes (rotated)
            num_stripes = 5
            for i in range(num_stripes + 1):
                t = i / num_stripes
                # Calculate points along the width edges
                p1 = lerp(world_points[0][0], world_points[3][0], t), lerp(world_points[0][1], world_points[3][1], t)
                p2 = lerp(world_points[1][0], world_points[2][0], t), lerp(world_points[1][1], world_points[2][1], t)
                pygame.draw.line(world_surface, BLACK, p1, p2, 2)

            hazards.append(("speedbreaker", hz_rect, hz_center))

    print(f"Environment created with {len(obstacles)} static obstacles and {len(hazards)} hazards.")
    return world_surface, obstacles, hazards


# --- Popup Manager (For general info) ---
class PopupManager:
    def __init__(self):
        self.popups = [] # List of tuples: (text_surface, rect, creation_time)

    def add_popup(self, text):
        # Avoid adding duplicate messages too quickly
        current_time = time.time()
        if self.popups and self.popups[-1][0].get_width() == popup_font.render(text, True, POPUP_TEXT).get_width():
             if current_time - self.popups[-1][2] < 0.5:
                 return

        text_surf = popup_font.render(text, True, POPUP_TEXT)
        text_rect = text_surf.get_rect()
        text_rect.centerx = SCREEN_WIDTH // 2
        text_rect.top = 20
        self.popups.append((text_surf, text_rect, current_time))

    def update(self):
        current_time = time.time()
        self.popups = [(s, r, t) for s, r, t in self.popups if current_time - t < POPUP_DURATION]

    def draw(self, surface):
        current_time = time.time()
        alpha_surface.fill((0,0,0,0)) # Clear alpha surface for popups

        popup_y_offset = 20 # Starting Y position
        for text_surf, text_rect_orig, creation_time in self.popups:
            elapsed = current_time - creation_time
            alpha = 255
            if POPUP_DURATION - elapsed < POPUP_FADE_TIME:
                alpha = int(255 * ((POPUP_DURATION - elapsed) / POPUP_FADE_TIME))
                alpha = max(0, min(255, alpha))

            if alpha > 0:
                text_rect = text_rect_orig.copy()
                text_rect.top = popup_y_offset

                bg_rect = text_rect.inflate(20, 10)
                bg_color = POPUP_BG[:3] + (int(POPUP_BG[3] * (alpha / 255.0)),)

                temp_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
                temp_surf.fill(bg_color)
                alpha_surface.blit(temp_surf, bg_rect.topleft)

                text_surf.set_alpha(alpha)
                alpha_surface.blit(text_surf, text_rect.topleft)

                popup_y_offset += bg_rect.height + 5

        surface.blit(alpha_surface, (0,0))


# --- Warning Manager (for Manual Failures) ---
class WarningManager:
    def __init__(self):
        self.warnings = [] # List of tuples: (text_surface, rect, creation_time)
        self.last_warning_time = 0

    def add_warning(self, text):
        current_time = time.time()
        if current_time - self.last_warning_time < WARNING_COOLDOWN:
            return

        text_surf = warning_font.render(text, True, WARNING_YELLOW)
        text_rect = text_surf.get_rect()
        text_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2) # Center screen
        self.warnings.append((text_surf, text_rect, current_time))
        self.last_warning_time = current_time

    def update(self):
        current_time = time.time()
        self.warnings = [(s, r, t) for s, r, t in self.warnings if current_time - t < WARNING_DURATION]

    def draw(self, surface):
        if not self.warnings:
            return

        # Draw only the most recent warning
        text_surf, text_rect, creation_time = self.warnings[-1]
        current_time = time.time()
        elapsed = current_time - creation_time

        # Flashing effect: Rapidly change alpha or visibility
        flash_interval = 0.2 # seconds
        visible = int(elapsed / flash_interval) % 2 == 0

        if visible:
            # Draw background rect using alpha surface for transparency
            alpha_surface.fill((0,0,0,0)) # Clear for warning
            bg_rect = text_rect.inflate(40, 20)
            temp_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            temp_surf.fill(WARNING_RED)
            pygame.draw.rect(temp_surf, WARNING_BORDER, temp_surf.get_rect(), width=5, border_radius=5)
            alpha_surface.blit(temp_surf, bg_rect.topleft)

            # Draw text on alpha surface
            alpha_surface.blit(text_surf, text_rect.topleft)

            # Blit alpha surface onto main screen
            surface.blit(alpha_surface, (0,0))


# --- Vehicle Class ---
class Vehicle:
    def __init__(self, start_pos, start_angle=0):
        self.x, self.y = start_pos
        self.angle = start_angle # Degrees, 0 = right, 90 = down
        self.speed = 0.0
        self.target_speed = VEHICLE_MAX_SPEED # Target speed for auto mode
        self.path = path_points
        self.current_path_index = 0
        self.state = "NORMAL" # AUTO state: NORMAL, EVADING_PLTHOLE, DECELERATING_SPEED_BREAKER
        self.evasion_target_offset = 0 # Lateral offset target during evasion (-1 left, 1 right, 0 none)
        self.current_lateral_offset = 0 # Current actual lateral offset from path center (approx)
        self.popup_manager = None
        self.warning_manager = None
        self.current_evasion_hazard = None # Store the hazard being evaded in AUTO
        self.control_mode = AUTO # Start in AUTO mode
        # Manual control inputs
        self.manual_steering = 0 # -1 for left, 1 for right, 0 for none
        self.manual_accel = 0 # 1 for accel, -1 for brake, 0 for none
        self.last_off_track_warning_time = 0 # Cooldown for off-track warning
        
        # Machine Learning and Data Collection
        self.data_collector = None
        self.recording = False
        self.frame_counter = 0
        self.previous_angle = start_angle
        self.previous_speed = 0.0
        
        # ML Models
        self.bc_model = None if not ML_AVAILABLE else BehaviorCloningModel()
        self.hazard_model = None if not ML_AVAILABLE else HazardDetectionModel()
        self.use_ml_control = False

    def set_popup_manager(self, manager):
        self.popup_manager = manager

    def set_warning_manager(self, manager):
        self.warning_manager = manager
        
    def set_data_collector(self, collector):
        self.data_collector = collector
        
    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self._show_popup("RECORDING STARTED")
        else:
            self._show_popup("RECORDING STOPPED")
        return self.recording
        
    def toggle_ml_control(self):
        if not ML_AVAILABLE:
            self._show_popup("ML FEATURES NOT AVAILABLE")
            return False
            
        self.use_ml_control = not self.use_ml_control
        if self.use_ml_control:
            self._show_popup("ML CONTROL ENABLED")
        else:
            self._show_popup("ML CONTROL DISABLED")
        return self.use_ml_control

    def _show_popup(self, text):
        if self.popup_manager:
            self.popup_manager.add_popup(text)

    def _show_warning(self, text):
        if self.warning_manager:
            self.warning_manager.add_warning(text)

    def update(self, dt, obstacles, hazards):
        """Main update method, routes to auto or manual logic."""
        # Save previous state for data collection
        previous_x, previous_y = self.x, self.y
        
        # ML-based control if enabled
        if self.use_ml_control and self.control_mode == AUTO and ML_AVAILABLE:
            self._update_ml(dt, obstacles, hazards)
        elif self.control_mode == AUTO:
            self._update_auto(dt, obstacles, hazards)
        else: # MANUAL
            self._update_manual(dt, obstacles, hazards)

        # Common physics update (apply speed, check collisions)
        self._apply_physics_and_collisions(dt, obstacles, hazards)
        
        # Record data if we're in recording mode
        if self.recording and self.data_collector:
            self.frame_counter += 1
            
            # Prepare vehicle state
            vehicle_state = {
                'x': self.x,
                'y': self.y,
                'angle': self.angle,
                'speed': self.speed,
                'lateral_offset': self.current_lateral_offset
            }
            
            # Prepare control inputs
            if self.control_mode == AUTO:
                control_inputs = {
                    'steering': angle_diff(self.previous_angle, self.angle),
                    'acceleration': (self.speed - self.previous_speed)
                }
            else:
                control_inputs = {
                    'steering': self.manual_steering,
                    'acceleration': self.manual_accel
                }
            
            # Get nearest hazard
            nearest_hazard = self._find_nearest_hazard(hazards)
            
            # Get path info
            path_info = {
                'target_x': self.path[self.current_path_index][0],
                'target_y': self.path[self.current_path_index][1],
                'segment_index': self.current_path_index
            }
            
            # Record driving frame
            self.data_collector.record_driving_frame(
                vehicle_state, 
                control_inputs, 
                nearest_hazard, 
                path_info
            )
            
            # Capture sensor data every 5 frames to reduce data size
            if self.frame_counter % 5 == 0:
                lidar_data = self._capture_lidar_data(obstacles, hazards)
                camera_data = self._capture_camera_view()
                image_filename = self.data_collector.save_camera_image(camera_data, self.frame_counter)
                self.data_collector.record_sensor_frame(lidar_data, image_filename)
            
            # Store previous values for calculating deltas
            self.previous_angle = self.angle
            self.previous_speed = self.speed

    def _update_ml(self, dt, obstacles, hazards):
        """ML-based driving logic using trained models"""
        if not self.bc_model or not ML_AVAILABLE:
            self._show_popup("ML MODEL NOT AVAILABLE - SWITCHING TO AUTO")
            self.use_ml_control = False
            self._update_auto(dt, obstacles, hazards)
            return
            
        # Prepare state vector for behavior cloning model
        state_vector = [
            self.x / WORLD_WIDTH,  # Normalize position
            self.y / WORLD_HEIGHT, 
            math.sin(math.radians(self.angle)),  # Use sin/cos for angle to avoid discontinuity
            math.cos(math.radians(self.angle)),
            self.speed / VEHICLE_MAX_SPEED,  # Normalize speed
            self.current_lateral_offset / LANE_WIDTH,  # Normalize offset
        ]
        
        # Get path information - relative target position
        target_index = self.current_path_index
        target_pos = self.path[target_index]
        next_target_index = (target_index + 1) % len(self.path)
        next_target_pos = self.path[next_target_index]
        
        # Add normalized path targets
        state_vector.extend([
            target_pos[0] / WORLD_WIDTH,
            target_pos[1] / WORLD_HEIGHT,
            next_target_pos[0] / WORLD_WIDTH,
            next_target_pos[1] / WORLD_HEIGHT
        ])
        
        # Use hazard detection model on camera data
        camera_view = self._capture_camera_view()
        if self.hazard_model:
            hazard_prediction = self.hazard_model.predict(camera_view)
            # hazard_prediction is [none_prob, pothole_prob, speedbreaker_prob]
            hazard_type = np.argmax(hazard_prediction)
            if hazard_type == 1:  # pothole
                self._show_popup("ML: Pothole detected!")
            elif hazard_type == 2:  # speedbreaker
                self._show_popup("ML: Speed breaker detected!")
        
        # Get behavior prediction from BC model
        prediction = self.bc_model.predict(state_vector)
        
        # Apply predicted steering and acceleration
        steering = prediction[0]  # -1 to 1
        acceleration = prediction[1]  # -1 to 1
        
        # Apply steering
        turn_rate = VEHICLE_TURN_RATE * dt * 60
        self.angle += steering * turn_rate
        self.angle %= 360
        
        # Apply acceleration
        if acceleration > 0:
            self.speed += VEHICLE_ACCEL * acceleration * dt * 60
            self.speed = min(self.speed, VEHICLE_MAX_SPEED)
        else:
            self.speed += VEHICLE_BRAKE_ACCEL * acceleration * dt * 60  # acceleration is negative
            self.speed = max(0, self.speed)

    def _find_nearest_hazard(self, hazards):
        """Find the nearest hazard and return its info"""
        nearest_hazard = None
        min_dist = float('inf')
        
        for hz_type, hz_rect, hz_center in hazards:
            dist = distance((self.x, self.y), hz_center)
            if dist < min_dist:
                min_dist = dist
                nearest_hazard = {
                    'type': hz_type,
                    'distance': dist,
                    'center': hz_center
                }
        
        return nearest_hazard

    def _capture_lidar_data(self, obstacles, hazards):
        """Simulate LiDAR readings"""
        lidar_data = []
        
        for i in range(LIDAR_NUM_RAYS):
            angle_offset = (i / (LIDAR_NUM_RAYS - 1) - 0.5) * LIDAR_ANGLE_SPREAD if LIDAR_NUM_RAYS > 1 else 0
            ray_angle_deg = self.angle + angle_offset
            ray_angle_rad = math.radians(ray_angle_deg)
            
            # Cast ray and find intersection
            ray_end = (self.x + LIDAR_RANGE * math.cos(ray_angle_rad),
                      self.y + LIDAR_RANGE * math.sin(ray_angle_rad))
            
            min_hit_dist = LIDAR_RANGE
            hit_point = None
            
            # Combine static and hazard rects for ray detection
            combined_obstacles = obstacles + [h[1] for h in hazards]
            
            for obs_rect in combined_obstacles:
                # Simple line-rect intersection using clipline
                try:
                    clipped_line = obs_rect.clipline((self.x, self.y), ray_end)
                    if clipped_line:
                        p1, p2 = clipped_line
                        # Find intersection point closest to vehicle
                        d1 = distance((self.x, self.y), p1)
                        d2 = distance((self.x, self.y), p2)
                        dist = min(d1, d2)
                        
                        if dist < min_hit_dist:
                            min_hit_dist = dist
                            hit_point = p1 if d1 < d2 else p2
                except TypeError:
                    pass # Ignore if clipline fails
            
            # Record distance or max range if no hit
            lidar_data.append((angle_offset, min_hit_dist))
        
        return lidar_data

    def _capture_camera_view(self):
        """Generate a simplified representation of what the camera sees"""
        # Create a simple image array of 50x100
        camera_view = np.zeros((50, 100))
        
        # In a real implementation, we'd extract a view from the rendered world
        # Here we'll just generate a very basic simulated view for ML purposes
        
        # Create a gradient that represents distance ahead
        for y in range(camera_view.shape[0]):
            camera_view[y, :] = y / camera_view.shape[0]
        
        # Add a "road" based on vehicle angle and lateral offset
        road_center = int(camera_view.shape[1] / 2 + self.current_lateral_offset * 5)
        road_width = int(LANE_WIDTH * 2 * 0.7)  # Scale for image size
        
        for x in range(max(0, road_center - road_width), min(camera_view.shape[1], road_center + road_width)):
            camera_view[:, x] += 0.3  # Brighten road area
        
        # Simulate any nearby hazards
        # (In a real implementation, this would be based on the actual rendered view)
        
        return camera_view

    def _update_auto(self, dt, obstacles, hazards):
        """ Autonomous driving logic """
        # --- Hazard Detection (AUTO specific reactions) ---
        detected_hazard = None
        min_hazard_dist = float('inf')
        vehicle_front_dist = VEHICLE_HEIGHT * 0.6 # Point slightly ahead of center

        # Calculate point ahead of vehicle for detection
        angle_rad = math.radians(self.angle)
        detect_point = (self.x + vehicle_front_dist * math.cos(angle_rad),
                        self.y + vehicle_front_dist * math.sin(angle_rad))

        for hz_type, hz_rect, hz_center in hazards:
            dist_to_center = distance(detect_point, hz_center)

            # Check if hazard is generally ahead
            vec_to_hazard = (hz_center[0] - self.x, hz_center[1] - self.y)
            vehicle_dir = (math.cos(angle_rad), math.sin(angle_rad))
            dot_product = vec_to_hazard[0] * vehicle_dir[0] + vec_to_hazard[1] * vehicle_dir[1]

            if dot_product > 0: # Hazard is generally in front
                if hz_type == "plothole" and dist_to_center < PLOTHOL_DETECT_RANGE:
                    if dist_to_center < min_hazard_dist:
                        min_hazard_dist = dist_to_center
                        detected_hazard = (hz_type, hz_rect, hz_center, dist_to_center)
                elif hz_type == "speedbreaker" and dist_to_center < SPEED_BREAKER_DETECT_RANGE:
                     if dist_to_center < min_hazard_dist:
                        min_hazard_dist = dist_to_center
                        detected_hazard = (hz_type, hz_rect, hz_center, dist_to_center)

        # --- State Machine (AUTO) ---
        if self.state == "NORMAL":
            self.target_speed = VEHICLE_MAX_SPEED # Ensure target speed is normal
            if detected_hazard:
                hz_type, hz_rect, hz_center, dist = detected_hazard
                if hz_type == "plothole":
                    self.state = "EVADING_PLTHOLE"
                    self.current_evasion_hazard = detected_hazard
                    # Decide evasion direction (simple: check which side of path center it is)
                    # Requires knowing path center line - approximate with path points
                    target_p1 = self.path[self.current_path_index]
                    target_p2 = self.path[(self.current_path_index + 1) % len(self.path)]
                    path_dx = target_p2[0] - target_p1[0]
                    path_dy = target_p2[1] - target_p1[1]
                    vec_to_hz_from_path = (hz_center[0] - target_p1[0], hz_center[1] - target_p1[1])
                    # Cross product to determine side (2D)
                    cross_product = path_dx * vec_to_hz_from_path[1] - path_dy * vec_to_hz_from_path[0]
                    self.evasion_target_offset = -1 if cross_product > 0 else 1 # Evade opposite side
                    self._show_popup(f"PLOTHOL DETECTED - EVADING {'RIGHT' if self.evasion_target_offset > 0 else 'LEFT'}")
                elif hz_type == "speedbreaker":
                    self.state = "DECELERATING_SPEED_BREAKER"
                    self.target_speed = SPEED_BREAKER_SLOW_SPEED
                    self._show_popup("SPEED BREAKER DETECTED - DECELERATING")

        elif self.state == "EVADING_PLTHOLE":
            # Check if evasion is complete (passed the hazard)
            if self.current_evasion_hazard:
                hz_type, hz_rect, hz_center, _ = self.current_evasion_hazard
                vec_to_hazard = (hz_center[0] - self.x, hz_center[1] - self.y)
                vehicle_dir = (math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
                dot_product = vec_to_hazard[0] * vehicle_dir[0] + vec_to_hazard[1] * vehicle_dir[1]
                # Check if hazard center is behind the vehicle's effective front
                if dot_product < -VEHICLE_HEIGHT * 0.3:
                    self.state = "NORMAL"
                    self.evasion_target_offset = 0 # Stop evading laterally
                    self.current_evasion_hazard = None
                    self._show_popup("EVASION COMPLETE - RETURNING")
            else: # Hazard lost or initial state error
                 self.state = "NORMAL"
                 self.evasion_target_offset = 0
                 self.current_evasion_hazard = None

        elif self.state == "DECELERATING_SPEED_BREAKER":
            # Check if passed or hazard lost
            passed = True
            if detected_hazard and detected_hazard[0] == "speedbreaker":
                # Check if still approaching/on it
                hz_type, hz_rect, hz_center, dist = detected_hazard
                vec_to_hazard = (hz_center[0] - self.x, hz_center[1] - self.y)
                vehicle_dir = (math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
                dot_product = vec_to_hazard[0] * vehicle_dir[0] + vec_to_hazard[1] * vehicle_dir[1]
                if dot_product > -hz_rect.height: # Still near or ahead of breaker center
                    passed = False
                    self.target_speed = SPEED_BREAKER_SLOW_SPEED # Ensure speed stays low

            if passed:
                self.state = "NORMAL"
                self.target_speed = VEHICLE_MAX_SPEED
                self._show_popup("PASSED SPEED BREAKER - ACCELERATING")

        # --- Speed Control (AUTO) ---
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + VEHICLE_ACCEL * dt * 60) # Scale by dt
        elif self.speed > self.target_speed:
            self.speed = max(self.target_speed, self.speed - VEHICLE_BRAKE_ACCEL * dt * 60)
        else: # Apply natural deceleration if at target speed but target is lower than max
             if self.target_speed < VEHICLE_MAX_SPEED:
                 self.speed = max(self.target_speed, self.speed - VEHICLE_DECEL * dt * 60)

        # --- Path Following & Steering (AUTO) ---
        target_node_idx = self.current_path_index
        next_node_idx = (target_node_idx + 1) % len(self.path)
        target_p1 = self.path[target_node_idx]
        target_p2 = self.path[next_node_idx]

        # Check distance to the *next* node to advance path index
        dist_to_next_node = distance((self.x, self.y), target_p2)
        if dist_to_next_node < LANE_WIDTH * 2.0: # Threshold to switch target segment
            self.current_path_index = next_node_idx
            target_node_idx = self.current_path_index
            next_node_idx = (target_node_idx + 1) % len(self.path)
            target_p1 = self.path[target_node_idx]
            target_p2 = self.path[next_node_idx]

        # Calculate target point slightly ahead on the current segment
        segment_dx = target_p2[0] - target_p1[0]
        segment_dy = target_p2[1] - target_p1[1]
        segment_len = distance(target_p1, target_p2)
        lookahead_dist = max(LANE_WIDTH, self.speed * 20) # Look further ahead at higher speeds

        # Project vehicle onto segment to find progress 'u'
        vec_car_from_p1 = (self.x - target_p1[0], self.y - target_p1[1])
        u = (vec_car_from_p1[0] * segment_dx + vec_car_from_p1[1] * segment_dy) / (segment_len**2 if segment_len > 0 else 1)
        u = max(0, min(1, u)) # Clamp progress along segment

        # Calculate target point on segment based on lookahead
        target_u = u + lookahead_dist / segment_len if segment_len > 0 else u
        target_u = min(target_u, 1.0) # Don't look beyond current segment end

        path_target_x = target_p1[0] + target_u * segment_dx
        path_target_y = target_p1[1] + target_u * segment_dy

        # Calculate lateral offset from the path center line (cross product method)
        path_normal_dx = -segment_dy / segment_len if segment_len > 0 else 0
        path_normal_dy = segment_dx / segment_len if segment_len > 0 else 0
        vec_car_from_closest = (self.x - (target_p1[0] + u * segment_dx), self.y - (target_p1[1] + u * segment_dy))
        self.current_lateral_offset = vec_car_from_closest[0] * path_normal_dx + vec_car_from_closest[1] * path_normal_dy

        # Adjust target point laterally for evasion or returning to center
        target_offset = self.evasion_target_offset * LANE_WIDTH * 0.6 # Target offset distance
        # Smoothly adjust lateral position using a P-controller (Proportional)
        lateral_error = target_offset - self.current_lateral_offset
        lateral_correction_speed = lateral_error * 0.05 # P-gain (adjust for stability)
        lateral_correction_speed = max(-VEHICLE_LANE_CHANGE_SPEED, min(VEHICLE_LANE_CHANGE_SPEED, lateral_correction_speed))

        # Apply lateral correction perpendicular to the path segment direction
        final_target_x = path_target_x + lateral_error * path_normal_dx # Aim to correct error directly
        final_target_y = path_target_y + lateral_error * path_normal_dy

        # Calculate angle to the final target point
        target_dx = final_target_x - self.x
        target_dy = final_target_y - self.y
        target_angle = math.degrees(math.atan2(target_dy, target_dx))

        # Calculate angle difference and steer smoothly
        angle_error = angle_diff(self.angle, target_angle)
        turn_rate_mult = 0.5 + 0.5 * (self.speed / VEHICLE_MAX_SPEED) # Scale turn rate by speed (less sensitive at low speed)
        turn_amount = angle_error * 0.1 # P-gain for steering (adjust for stability)
        max_turn = (VEHICLE_TURN_RATE if self.state != "EVADING_PLTHOLE" else VEHICLE_EVADE_TURN_RATE) * turn_rate_mult * dt * 60
        turn_amount = max(-max_turn, min(max_turn, turn_amount))

        self.angle = (self.angle + turn_amount) % 360


    def _update_manual(self, dt, obstacles, hazards):
        """ Manual driving logic and failure checks """
        # --- Manual Control Input Application ---
        # Steering
        if self.manual_steering != 0:
            manual_turn_rate = VEHICLE_TURN_RATE * MANUAL_TURN_MULT
            # Scale turn rate by speed, less sensitive at low speed
            turn_scale = 0.3 + 0.7 * min(1.0, self.speed / (VEHICLE_MAX_SPEED * 0.5))
            self.angle += self.manual_steering * manual_turn_rate * turn_scale * dt * 60
            self.angle %= 360

        # Acceleration/Braking
        if self.manual_accel > 0: # Accelerate
            self.speed += VEHICLE_ACCEL * MANUAL_ACCEL_MULT * dt * 60
            self.speed = min(self.speed, VEHICLE_MAX_SPEED * MANUAL_MAX_SPEED_MULT)
        elif self.manual_accel < 0: # Brake
            self.speed -= VEHICLE_BRAKE_ACCEL * MANUAL_BRAKE_MULT * dt * 60
            self.speed = max(0, self.speed)
        else: # No input - natural deceleration
            self.speed -= VEHICLE_DECEL * dt * 60
            self.speed = max(0, self.speed)

        # --- Manual Failure Checks ---
        vehicle_rect = self.get_rect() # Use a simplified rect for collision checks
        vehicle_center = (self.x, self.y)

        # 1. Off-track check
        min_dist_to_path_segment = float('inf')
        for i in range(len(self.path)):
            p1 = self.path[i]
            p2 = self.path[(i + 1) % len(self.path)]
            dist = point_segment_distance(self.x, self.y, p1[0], p1[1], p2[0], p2[1])
            min_dist_to_path_segment = min(min_dist_to_path_segment, dist)

        if min_dist_to_path_segment > MANUAL_OFF_TRACK_THRESHOLD:
            current_time = time.time()
            # Add cooldown to prevent spamming off-track warnings
            if current_time - self.last_off_track_warning_time > WARNING_COOLDOWN * 2:
                 self._show_warning("CAUTION: OFF TRACK!")
                 self.last_off_track_warning_time = current_time
                 # Optional: Add slight speed penalty for going off track
                 self.speed *= 0.95

        # 2. Hazard Collision Checks
        for hz_type, hz_rect, hz_center in hazards:
            # Use circle collision for potholes, rect for breakers (more accurate)
            collided = False
            if hz_type == "plothole":
                radius = hz_rect.width / 2
                if distance(vehicle_center, hz_center) < radius + VEHICLE_WIDTH / 2: # Circle collision approx
                    collided = True
            elif hz_type == "speedbreaker":
                 # Use rotated rect collision check (more complex)
                 # Simplified: Use bounding box collision for now
                 if vehicle_rect.colliderect(hz_rect):
                     collided = True # Approximate check

            if collided:
                if hz_type == "plothole":
                    self._show_warning("COLLISION DETECTED!")
                    self.speed *= 0.6 # Significant speed penalty
                elif hz_type == "speedbreaker":
                    if self.speed > MANUAL_SPEED_BREAKER_THRESHOLD:
                        self._show_warning(f"SPEEDBREAKER VIOLATION! ({self.speed:.1f} > {MANUAL_SPEED_BREAKER_THRESHOLD:.1f})")
                        self.speed *= 0.7 # Speed penalty


    def _apply_physics_and_collisions(self, dt, obstacles, hazards):
        """ Applies movement based on speed/angle and handles collisions """
        # Calculate potential movement
        angle_rad = math.radians(self.angle)
        move_x = self.speed * math.cos(angle_rad) * dt * 60 # Scale by dt
        move_y = self.speed * math.sin(angle_rad) * dt * 60

        # --- Apply Lateral Correction (AUTO mode only) ---
        # This is a simplified way to handle the lateral offset correction calculated in _update_auto
        if self.control_mode == AUTO and self.state == "EVADING_PLTHOLE":
            # Calculate the desired lateral velocity to correct the offset
            target_offset = self.evasion_target_offset * LANE_WIDTH * 0.6
            lateral_error = target_offset - self.current_lateral_offset
            # P-controller for lateral speed (adjust gain as needed)
            lateral_correction_speed = lateral_error * 0.1 # Gain value
            # Clamp correction speed
            lateral_correction_speed = max(-VEHICLE_LANE_CHANGE_SPEED, min(VEHICLE_LANE_CHANGE_SPEED, lateral_correction_speed))

            # Apply this speed perpendicular to the vehicle's heading
            perp_angle_rad = math.radians(self.angle + 90)
            lat_move_x = lateral_correction_speed * math.cos(perp_angle_rad) * dt * 60
            lat_move_y = lateral_correction_speed * math.sin(perp_angle_rad) * dt * 60
            move_x += lat_move_x
            move_y += lat_move_y
        elif self.control_mode == AUTO and self.state == "NORMAL" and abs(self.current_lateral_offset) > 1.0:
             # Gently return to center if offset exists in NORMAL state
             target_offset = 0
             lateral_error = target_offset - self.current_lateral_offset
             lateral_correction_speed = lateral_error * 0.05 # Slower return gain
             lateral_correction_speed = max(-VEHICLE_LANE_CHANGE_SPEED * 0.5, min(VEHICLE_LANE_CHANGE_SPEED * 0.5, lateral_correction_speed))
             perp_angle_rad = math.radians(self.angle + 90)
             lat_move_x = lateral_correction_speed * math.cos(perp_angle_rad) * dt * 60
             lat_move_y = lateral_correction_speed * math.sin(perp_angle_rad) * dt * 60
             move_x += lat_move_x
             move_y += lat_move_y


        # Store potential next position
        next_x = self.x + move_x
        next_y = self.y + move_y

        # Collision check with static obstacles
        # Create a slightly larger rect for collision checking based on potential next pos
        vehicle_check_rect = pygame.Rect(0, 0, VEHICLE_WIDTH, VEHICLE_HEIGHT)
        vehicle_check_rect.center = (next_x, next_y)
        # TODO: Implement rotated rectangle collision for better accuracy

        collision_detected = False
        for obs_rect in obstacles:
            if vehicle_check_rect.colliderect(obs_rect): # Simple AABB check
                collision_detected = True
                break

        if collision_detected:
            # Collision response: Stop movement, maybe slight bounce
            self.speed *= 0.1 # Drastic speed reduction
            # Don't update position: self.x, self.y remain unchanged this frame
            if self.control_mode == MANUAL:
                self._show_warning("COLLISION DETECTED!")
            else:
                self._show_popup("AUTO: Obstacle Collision!")
        else:
            # Apply movement if no collision
            self.x = next_x
            self.y = next_y

        # Keep vehicle within world bounds (simple clamp)
        self.x = max(VEHICLE_WIDTH / 2, min(WORLD_WIDTH - VEHICLE_WIDTH / 2, self.x))
        self.y = max(VEHICLE_HEIGHT / 2, min(WORLD_HEIGHT - VEHICLE_HEIGHT / 2, self.y))


    def get_rect(self):
        """ Returns an axis-aligned bounding box for the vehicle.
            Note: For accurate collision, rotated collision detection is needed. """
        # Create a rect centered on the vehicle's position
        # Use the larger dimension for safety in AABB checks
        max_dim = max(VEHICLE_WIDTH, VEHICLE_HEIGHT)
        rect = pygame.Rect(0, 0, max_dim, max_dim)
        rect.center = (self.x, self.y)
        return rect

    def draw(self, surface, camera_offset, obstacles=None, hazards=None):
        """Draws the vehicle and its sensors."""
        screen_pos = world_to_screen((self.x, self.y), camera_offset)

        # Create a vehicle surface to rotate (more detailed)
        vehicle_surf = pygame.Surface((VEHICLE_HEIGHT, VEHICLE_WIDTH), pygame.SRCALPHA)
        vehicle_surf.fill((0,0,0,0))
        body_rect = pygame.Rect(0, 0, VEHICLE_HEIGHT, VEHICLE_WIDTH) # Base rect before rotation

        # Main body
        pygame.draw.rect(vehicle_surf, VEHICLE_RED, body_rect.inflate(-2, -2), border_radius=6)
        pygame.draw.rect(vehicle_surf, VEHICLE_DARK_RED, body_rect.inflate(-2, -2), width=3, border_radius=6)

        # Windshield (front)
        windshield_poly = [
            (body_rect.width * 0.55, body_rect.top + 3),
            (body_rect.width * 0.85, body_rect.top + 3),
            (body_rect.width * 0.75, body_rect.centery - 3),
            (body_rect.width * 0.6, body_rect.centery - 3),
        ]
        pygame.draw.polygon(vehicle_surf, VEHICLE_WINDOW, windshield_poly)
        pygame.draw.lines(vehicle_surf, BLACK, True, windshield_poly, 1)

         # Rear window (smaller)
        rear_window_poly = [
            (body_rect.width * 0.1, body_rect.top + 5),
            (body_rect.width * 0.3, body_rect.top + 5),
            (body_rect.width * 0.25, body_rect.centery - 5),
            (body_rect.width * 0.15, body_rect.centery - 5),
        ]
        pygame.draw.polygon(vehicle_surf, VEHICLE_WINDOW, rear_window_poly)


        # Headlights
        headlight_size = 4
        pygame.draw.circle(vehicle_surf, VEHICLE_LIGHT, (body_rect.right - headlight_size - 3, body_rect.top + headlight_size + 2), headlight_size)
        pygame.draw.circle(vehicle_surf, VEHICLE_LIGHT, (body_rect.right - headlight_size - 3, body_rect.bottom - headlight_size - 2), headlight_size)
        # Taillights (dimmer red)
        taillight_size = 3
        pygame.draw.circle(vehicle_surf, VEHICLE_DARK_RED, (body_rect.left + taillight_size + 2, body_rect.top + taillight_size + 2), taillight_size)
        pygame.draw.circle(vehicle_surf, VEHICLE_DARK_RED, (body_rect.left + taillight_size + 2, body_rect.bottom - taillight_size - 2), taillight_size)


        # Rotate the vehicle surface
        rotated_surf = pygame.transform.rotate(vehicle_surf, -self.angle) # Pygame rotates counter-clockwise
        rotated_rect = rotated_surf.get_rect(center=screen_pos)

        # Blit the rotated vehicle
        surface.blit(rotated_surf, rotated_rect.topleft)

        # --- Draw Sensors (on alpha surface for blending) ---
        # Optional visualization - can be performance intensive
        draw_sensors = True # Set to False to disable sensor drawing
        if draw_sensors and obstacles is not None and hazards is not None:
            alpha_surface.fill((0,0,0,0)) # Clear alpha surface each frame

            vehicle_center_world = (self.x, self.y)
            vehicle_center_screen = screen_pos

            # Combine static and hazard rects for sensor detection
            combined_obstacles_for_sensors = obstacles + [h[1] for h in hazards]

            # --- Mock LiDAR ---
            lidar_hit_points_screen = [] # Store screen positions of hits

            for i in range(LIDAR_NUM_RAYS):
                angle_offset = (i / (LIDAR_NUM_RAYS - 1) - 0.5) * LIDAR_ANGLE_SPREAD if LIDAR_NUM_RAYS > 1 else 0
                ray_angle_deg = self.angle + angle_offset
                ray_angle_rad = math.radians(ray_angle_deg)
                ray_end_world = (vehicle_center_world[0] + LIDAR_RANGE * math.cos(ray_angle_rad),
                                 vehicle_center_world[1] + LIDAR_RANGE * math.sin(ray_angle_rad))

                min_hit_dist_sq = LIDAR_RANGE**2
                hit_point_world = None

                for obs_rect in combined_obstacles_for_sensors:
                    # Simple line-rect intersection using clipline
                    try:
                        clipped_line = obs_rect.clipline(vehicle_center_world, ray_end_world)
                        if clipped_line:
                            p1, p2 = clipped_line
                            # Find intersection point closest to vehicle
                            d1_sq = distance(vehicle_center_world, p1)**2
                            d2_sq = distance(vehicle_center_world, p2)**2
                            dist_sq = min(d1_sq, d2_sq)
                            intersect_point = p1 if d1_sq < d2_sq else p2

                            if dist_sq < min_hit_dist_sq:
                                min_hit_dist_sq = dist_sq
                                hit_point_world = intersect_point
                    except TypeError:
                        pass # Ignore if clipline fails

                if hit_point_world:
                    lidar_hit_points_screen.append(world_to_screen(hit_point_world, camera_offset))
                # else: # Draw ray to max range if no hit (optional)
                #     ray_end_screen = world_to_screen(ray_end_world, camera_offset)
                #     pygame.draw.line(alpha_surface, SENSOR_LIDAR_COLOR + (30,), vehicle_center_screen, ray_end_screen, 1)


            # Draw LiDAR hits
            for point_screen in lidar_hit_points_screen:
                pygame.draw.circle(alpha_surface, SENSOR_LIDAR_COLOR, point_screen, 3)

            # --- Mock Camera FOV ---
            fov_points_world = [vehicle_center_world]
            for angle_offset in [-CAMERA_FOV_ANGLE / 2, CAMERA_FOV_ANGLE / 2]:
                 angle_rad = math.radians(self.angle + angle_offset)
                 fov_points_world.append((vehicle_center_world[0] + CAMERA_RANGE * math.cos(angle_rad),
                                          vehicle_center_world[1] + CAMERA_RANGE * math.sin(angle_rad)))

            fov_points_screen = [world_to_screen(p, camera_offset) for p in fov_points_world]

            if len(fov_points_screen) >= 3:
                pygame.draw.polygon(alpha_surface, SENSOR_CAMERA_COLOR, fov_points_screen)

            # Blit the alpha surface onto the main screen
            surface.blit(alpha_surface, (0,0))


# --- Main Game Loop ---
def main():
    running = True
    clock = pygame.time.Clock()

    print("Creating simulation environment...")
    world_surface, obstacles, hazards = create_environment(WORLD_WIDTH, WORLD_HEIGHT)

    # Determine start angle based on the first path segment
    start_angle = 0
    if len(path_points) > 1:
        dx = path_points[1][0] - path_points[0][0]
        dy = path_points[1][1] - path_points[0][1]
        start_angle = math.degrees(math.atan2(dy, dx))

    vehicle = Vehicle(path_points[0], start_angle=start_angle)

    popup_manager = PopupManager()
    warning_manager = WarningManager()
    vehicle.set_popup_manager(popup_manager)
    vehicle.set_warning_manager(warning_manager)
    
    # Create data collector and setup for ML
    data_collector = DataCollector()
    vehicle.set_data_collector(data_collector)
    
    # Initial camera position centered on vehicle
    camera_offset_x = vehicle.x - SCREEN_WIDTH // 2
    camera_offset_y = vehicle.y - SCREEN_HEIGHT // 2

    print("Starting simulation loop...")
    print(f"Controls: Arrows = Move (Manual), M = Switch Mode, R = Toggle Recording, L = Toggle ML (if available), ESC = Quit")
    vehicle._show_popup("AUTOMATIC MODE ENGAGED") # Initial mode message
    
    # Define new control keys
    KEY_RECORD = pygame.K_r
    KEY_ML_TOGGLE = pygame.K_l
    KEY_TRAIN = pygame.K_t

    while running:
        # Calculate delta time
        dt = clock.tick(60) / 1000.0 # Time since last frame in seconds
        if dt > 0.1: dt = 0.1 # Cap delta time to prevent physics glitches

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Mode Switching
                if event.key == KEY_MODE_SWITCH:
                    if vehicle.control_mode == AUTO:
                        vehicle.control_mode = MANUAL
                        vehicle._show_popup("MANUAL MODE ENGAGED")
                        # Reset auto state variables if needed
                        vehicle.state = "NORMAL"
                        vehicle.target_speed = vehicle.speed # Keep current speed initially
                    else:
                        vehicle.control_mode = AUTO
                        vehicle._show_popup("AUTOMATIC MODE ENGAGED")
                        # Reset manual inputs
                        vehicle.manual_steering = 0
                        vehicle.manual_accel = 0
                        # Reset auto state machine and target speed
                        vehicle.state = "NORMAL"
                        vehicle.target_speed = VEHICLE_MAX_SPEED # Resume normal auto speed target
                        vehicle.evasion_target_offset = 0 # Ensure no residual evasion target
                
                # Toggle Recording
                if event.key == KEY_RECORD:
                    is_recording = vehicle.toggle_recording()
                    if not is_recording:
                        # Save data if recording was stopped
                        session_dir = data_collector.save_datasets()
                        vehicle._show_popup(f"Data saved to {session_dir}")
                
                # Toggle ML control
                if event.key == KEY_ML_TOGGLE and ML_AVAILABLE:
                    vehicle.toggle_ml_control()
                
                # Train ML model
                if event.key == KEY_TRAIN and ML_AVAILABLE:
                    # If recording is active, stop it first
                    if vehicle.recording:
                        vehicle.toggle_recording()
                        data_collector.save_datasets()
                    
                    # Check if there's any data to train from
                    if os.path.exists("datasets"):
                        session_dirs = [os.path.join("datasets", d) for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
                        if session_dirs:
                            vehicle._show_popup("Training ML models... (may take some time)")
                            # Use most recent session by default
                            latest_session = max(session_dirs, key=os.path.getmtime)
                            
                            # Train the models
                            if vehicle.bc_model:
                                try:
                                    vehicle.bc_model.train(latest_session)
                                    vehicle._show_popup("Behavior Cloning model trained!")
                                except Exception as e:
                                    vehicle._show_popup(f"BC training error: {str(e)[:20]}")
                            
                            if vehicle.hazard_model:
                                try:
                                    vehicle.hazard_model.train(latest_session)
                                    vehicle._show_popup("Hazard Detection model trained!")
                                except Exception as e:
                                    vehicle._show_popup(f"HD training error: {str(e)[:20]}")
                        else:
                            vehicle._show_popup("No training data found!")
                    else:
                        vehicle._show_popup("No datasets directory found!")

            # Manual Control Input (Key Down/Up) - Only active in MANUAL mode
            if vehicle.control_mode == MANUAL:
                if event.type == pygame.KEYDOWN:
                    if event.key == KEY_ACCEL:
                        vehicle.manual_accel = 1
                    elif event.key == KEY_BRAKE:
                        vehicle.manual_accel = -1
                    elif event.key == KEY_LEFT:
                        vehicle.manual_steering = -1
                    elif event.key == KEY_RIGHT:
                        vehicle.manual_steering = 1
                elif event.type == pygame.KEYUP:
                    if event.key == KEY_ACCEL and vehicle.manual_accel > 0:
                        vehicle.manual_accel = 0
                    elif event.key == KEY_BRAKE and vehicle.manual_accel < 0:
                        vehicle.manual_accel = 0
                    elif event.key == KEY_LEFT and vehicle.manual_steering < 0:
                        vehicle.manual_steering = 0
                    elif event.key == KEY_RIGHT and vehicle.manual_steering > 0:
                        vehicle.manual_steering = 0

        # --- Update ---
        vehicle.update(dt, obstacles, hazards)
        popup_manager.update()
        warning_manager.update()

        # Update camera smoothly towards vehicle position
        target_camera_x = vehicle.x - SCREEN_WIDTH // 2
        target_camera_y = vehicle.y - SCREEN_HEIGHT // 2
        # Apply lerp smoothing
        camera_offset_x = lerp(camera_offset_x, target_camera_x, CAMERA_LAG * dt * 60) # Scale lag by dt
        camera_offset_y = lerp(camera_offset_y, target_camera_y, CAMERA_LAG * dt * 60)
        camera_offset = (camera_offset_x, camera_offset_y)

        # --- Drawing ---
        screen.fill(BLACK) # Clear screen

        # Draw world surface (background) shifted by camera offset
        screen.blit(world_surface, (-camera_offset_x, -camera_offset_y))

        # Draw vehicle and its sensors
        vehicle.draw(screen, camera_offset, obstacles, hazards)

        # Draw popups (general info)
        popup_manager.draw(screen)

        # Draw Warnings (manual failures - drawn on top)
        warning_manager.draw(screen)

        # Draw HUD/Debug info
        mode_color = SENSOR_LIDAR_COLOR if vehicle.control_mode == AUTO else WARNING_YELLOW
        hud_text = [
            f"FPS: {clock.get_fps():.1f}",
            f"Mode: {vehicle.control_mode}",
            f"Speed: {vehicle.speed:.1f}",
            f"Pos: ({vehicle.x:.0f}, {vehicle.y:.0f})",
            f"Angle: {vehicle.angle:.1f}",
        ]
        if vehicle.control_mode == AUTO:
             hud_text.append(f"Auto State: {vehicle.state}")
             hud_text.append(f"Lat Offset: {vehicle.current_lateral_offset:.1f} (Tgt: {vehicle.evasion_target_offset * LANE_WIDTH * 0.6:.1f})")
        else: # Manual Mode HUD
             hud_text.append(f"Manual Input: A={vehicle.manual_accel}, S={vehicle.manual_steering}")

        # Add ML and recording status
        if ML_AVAILABLE:
            ml_status = "ENABLED" if vehicle.use_ml_control else "DISABLED"
            hud_text.append(f"ML Control: {ml_status}")
        
        rec_status = "RECORDING" if vehicle.recording else "NOT RECORDING"
        hud_text.append(f"Data Collection: {rec_status}")

        for i, line in enumerate(hud_text):
            color = mode_color if "Mode:" in line else WHITE # Highlight mode line
            try:
                text_surf = default_font.render(line, True, color)
                screen.blit(text_surf, (10, 10 + i * 20))
            except Exception as e:
                print(f"Error rendering HUD text: {e}") # Catch font rendering errors

        # Add control hint
        try:
            hint_1 = f"[{pygame.key.name(KEY_MODE_SWITCH).upper()}] Switch Mode | [{pygame.key.name(KEY_RECORD).upper()}] Toggle Recording"
            hint_2 = f"[{pygame.key.name(KEY_ML_TOGGLE).upper()}] Toggle ML | [{pygame.key.name(KEY_TRAIN).upper()}] Train ML"
            hint_surf_1 = default_font.render(hint_1, True, WHITE)
            hint_surf_2 = default_font.render(hint_2, True, WHITE)
            screen.blit(hint_surf_1, (10, SCREEN_HEIGHT - 45))
            screen.blit(hint_surf_2, (10, SCREEN_HEIGHT - 25))
        except Exception as e:
             print(f"Error rendering hint text: {e}")

        # --- Display Update ---
        pygame.display.flip()

    # When exiting, save any remaining data if recording
    if vehicle.recording:
        data_collector.save_datasets()
    
    print("Exiting simulation.")
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n--- UNHANDLED EXCEPTION ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        # Optionally print traceback
        import traceback
        traceback.print_exc()
        print(f"---------------------------\n")
        pygame.quit()
        sys.exit(1)