import pygame
import sys
import os
import math
import random
import threading
import socket
import pickle
import time
from collections import deque
from layout import LayoutManager

# Import game components
from environment import ScoponeEnvMA
from actions import encode_action, decode_action
from main import DQNAgent

# Constants
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GREEN = (0, 100, 0)
TABLE_GREEN = (26, 105, 66)
HIGHLIGHT_BLUE = (0, 120, 215)
HIGHLIGHT_RED = (220, 53, 69)
GOLD = (255, 215, 0)
DARK_BLUE = (20, 51, 104)
LIGHT_BLUE = (100, 149, 237)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

# Card dimensions
CARD_WIDTH = 80
CARD_HEIGHT = 120

# Suit symbols mapping (though we'll use images)
SUIT_SYMBOLS = {
    'denari': '♦',
    'coppe': '♥',
    'spade': '♠',
    'bastoni': '♣'
}

# Rank names for display
RANK_NAMES = {
    1: "Asso",
    2: "Due",
    3: "Tre",
    4: "Quattro",
    5: "Cinque",
    6: "Sei",
    7: "Sette",
    8: "Otto",
    9: "Nove",
    10: "Dieci"
}

# Suit names for display
SUIT_NAMES = {
    'denari': 'denari',
    'coppe': 'coppe',
    'spade': 'spade',
    'bastoni': 'bastoni'
}

# Card mapping helper function
def format_card(card):
    """
    Converts a (rank, suit) tuple to the image filename format
    Example: (1, 'denari') -> "01_Asso_di_denari"
    """
    rank, suit = card
    if suit == 'denari':
        num = rank
    elif suit == 'coppe':
        num = rank + 10
    elif suit == 'spade':
        num = rank + 20
    elif suit == 'bastoni':
        num = rank + 30
    else:
        num = rank
    prefix = f"{num:02d}"
    return f"{prefix}_{RANK_NAMES[rank]}_di_{SUIT_NAMES[suit]}"

def get_local_ip():
    """Ottiene l'indirizzo IP locale della macchina"""
    try:
        # Crea un socket temporaneo per ottenere l'IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non serve una connessione reale
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"  # Fallback su localhost

class LoadingAnimation:
    """Animazione di caricamento per quando si caricano i bot AI"""
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.start_time = pygame.time.get_ticks()
        self.dots = 0
        self.dots_timer = 0
        self.rotation = 0
        
    def update(self):
        # Aggiorna la rotazione dell'animazione
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time
        self.rotation = (elapsed / 10) % 360
        
        # Aggiorna i puntini animati
        if current_time - self.dots_timer > 500:  # Cambia puntini ogni 500ms
            self.dots = (self.dots + 1) % 4
            self.dots_timer = current_time
    
    def draw(self, surface, message=None):
        # Crea un overlay semi-trasparente
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Nero semi-trasparente
        surface.blit(overlay, (0, 0))
        
        # Disegna il cerchio rotante
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        radius = min(self.screen_width, self.screen_height) * 0.1
        
        # Disegna archi di cerchio rotanti
        for i in range(8):
            angle = self.rotation + i * 45
            start_angle = math.radians(angle - 20)
            end_angle = math.radians(angle + 20)
            
            # Calcola i punti dell'arco
            color_intensity = 255 - (i * 20)  # Sfuma il colore
            color = (color_intensity, color_intensity, 255)  # Blu che sfuma
            
            # Disegna un arco spesso
            pygame.draw.arc(surface, color, 
                           (center_x - radius, center_y - radius, radius*2, radius*2),
                           start_angle, end_angle, width=int(radius*0.2))
        
        # Disegna il messaggio di caricamento
        font = pygame.font.SysFont(None, int(self.screen_height * 0.04))
        dots_text = "." * self.dots
        if message is None:
            message = "Caricamento bot AI in corso"
        text = f"{message}{dots_text}"
        text_surf = font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(center_x, center_y + radius * 1.5))
        surface.blit(text_surf, text_rect)
        
class Button:
    """Enhanced button with hover and click effects"""
    def __init__(self, x, y, width, height, text, color, text_color, 
                 hover_color=None, font_size=24, border_radius=5):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = hover_color or self._brighten_color(color, 20)
        self.font = pygame.font.SysFont(None, font_size)
        self.border_radius = border_radius
        self.hovered = False
        self.clicked = False
        self.disabled = False
    
    def _brighten_color(self, color, amount):
        r, g, b = color
        return (max(0, min(255, r + amount)), 
                max(0, min(255, g + amount)), 
                max(0, min(255, b + amount)))
    
    def draw(self, surface):
        # Determine current color based on state
        current_color = self.color
        if self.disabled:
            current_color = GRAY
        elif self.clicked:
            current_color = self._brighten_color(self.color, -20)
        elif self.hovered:
            current_color = self.hover_color
        
        # Draw button with rounded corners
        pygame.draw.rect(surface, current_color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, self._brighten_color(current_color, -50), self.rect, 
                         2, border_radius=self.border_radius)
        
        # Render text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def update(self, events):
        # Reset click state
        self.clicked = False
        
        if self.disabled:
            return False
            
        # Check for hover
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)
        
        # Check for click
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.hovered:
                    self.clicked = True
                    return True
        
        return False
    
    def is_clicked(self, pos):
        if self.disabled:
            return False
        return self.rect.collidepoint(pos)

class CardAnimation:
    """Class for card movement animations"""
    def __init__(self, card, start_pos, end_pos, duration, delay=0, 
                 scale_start=1.0, scale_end=1.0, rotation_start=0, rotation_end=0):
        self.card = card
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.delay = delay
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.rotation_start = rotation_start
        self.rotation_end = rotation_end
        
        self.current_frame = -delay
        self.done = False
    
    def update(self):
        if self.current_frame < 0:
            self.current_frame += 1
            return False
            
        if self.current_frame >= self.duration:
            self.done = True
            return True
            
        self.current_frame += 1
        return False
    
    def get_current_pos(self):
        if self.current_frame < 0:
            return self.start_pos
            
        if self.current_frame >= self.duration:
            return self.end_pos
            
        # Ease-out function for smooth animation
        progress = self.current_frame / self.duration
        ease_progress = 1 - (1 - progress) * (1 - progress)
        
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * ease_progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * ease_progress
        
        return (x, y)
    
    def get_current_scale(self):
        if self.current_frame < 0:
            return self.scale_start
            
        if self.current_frame >= self.duration:
            return self.scale_end
            
        progress = self.current_frame / self.duration
        return self.scale_start + (self.scale_end - self.scale_start) * progress
    
    def get_current_rotation(self):
        if self.current_frame < 0:
            return self.rotation_start
            
        if self.current_frame >= self.duration:
            return self.rotation_end
            
        progress = self.current_frame / self.duration
        return self.rotation_start + (self.rotation_end - self.rotation_start) * progress

class ResourceManager:
    """Manages and caches game resources like images and sounds"""
    def __init__(self):
        self.card_images = {}
        self.card_backs = {}
        self.background = None
        self.table_texture = None
        self.ui_elements = {}
        self.sounds = {}
        self.sound_enabled = False
        
    def load_resources(self):
        """Load all game resources"""
        self.load_card_images()
        self.load_backgrounds()
        self.load_ui_elements()
        self.try_load_sounds()
        
    def rescale_card_images(self, card_width, card_height):
        """Rescale all card images to the new size"""
        # Store original images if we haven't already
        if not hasattr(self, 'original_card_images'):
            self.original_card_images = self.card_images.copy()
            self.original_card_backs = self.card_backs.copy()
        
        # Rescale all card images
        for key, original_img in self.original_card_images.items():
            self.card_images[key] = pygame.transform.scale(original_img, (card_width, card_height))
        
        # Rescale card backs
        for team_id, original_back in self.original_card_backs.items():
            self.card_backs[team_id] = pygame.transform.scale(original_back, (card_width, card_height))

    def load_card_images(self, folder="assets"):
        """Load card images from the assets folder"""
        possible_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        possible_suits = ['denari', 'coppe', 'spade', 'bastoni']
        
        # Store the original images at their native resolution
        self.original_card_images = {}
        self.original_card_backs = {}
        
        # Load card backs first (one for each team)
        try:
            back_blue = pygame.image.load(os.path.join(folder, "card_back_blue.jpg"))
            back_red = pygame.image.load(os.path.join(folder, "card_back_red.jpg"))
            self.original_card_backs[0] = back_blue
            self.original_card_backs[1] = back_red
            self.card_backs[0] = pygame.transform.scale(back_blue, (CARD_WIDTH, CARD_HEIGHT))
            self.card_backs[1] = pygame.transform.scale(back_red, (CARD_WIDTH, CARD_HEIGHT))
        except:
            # Create default card backs if images not found
            back_blue = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
            back_blue.fill(LIGHT_BLUE)
            back_red = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
            back_red.fill(HIGHLIGHT_RED)
            self.original_card_backs[0] = back_blue
            self.original_card_backs[1] = back_red
            self.card_backs[0] = back_blue
            self.card_backs[1] = back_red
        
        # Load all card images
        for r in possible_ranks:
            for s in possible_suits:
                key = (r, s)
                filename_key = format_card(key)
                filename = os.path.join(folder, filename_key + ".jpg")
                
                try:
                    img = pygame.image.load(filename)
                    self.original_card_images[key] = img
                    self.card_images[key] = pygame.transform.scale(img, (CARD_WIDTH, CARD_HEIGHT))
                except:
                    # Create a placeholder if image not found
                    card_img = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
                    card_img.fill(WHITE)
                    pygame.draw.rect(card_img, BLACK, (0, 0, CARD_WIDTH, CARD_HEIGHT), 2)
                    
                    # Draw rank and suit
                    font = pygame.font.SysFont(None, 24)
                    rank_text = str(r) if r > 1 else "A"
                    suit_symbol = SUIT_SYMBOLS[s]
                    
                    rank_surf = font.render(rank_text, True, BLACK)
                    suit_surf = font.render(suit_symbol, True, 
                                        HIGHLIGHT_RED if s in ['denari', 'coppe'] else BLACK)
                    
                    card_img.blit(rank_surf, (5, 5))
                    card_img.blit(suit_surf, (CARD_WIDTH - 20, 5))
                    
                    self.original_card_images[key] = card_img.copy()
                    self.card_images[key] = card_img

    def load_backgrounds(self, folder="assets"):
        """Load background images and textures"""
        try:
            bg = pygame.image.load(os.path.join(folder, "background.jpg"))
            self.original_background = bg  # Store the original unscaled image
            self.background = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except:
            # Create a default background if image not found
            self.original_background = pygame.Surface((1024, 768))
            self.original_background.fill(DARK_BLUE)
            self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.background.fill(DARK_BLUE)
        
        try:
            table = pygame.image.load(os.path.join(folder, "table_texture.jpg"))
            self.original_table_texture = table
            self.table_texture = pygame.transform.scale(table, (750, 450))
        except:
            # Create a default table texture if image not found
            self.original_table_texture = pygame.Surface((750, 450))
            self.original_table_texture.fill(TABLE_GREEN)
            self.table_texture = pygame.Surface((750, 450))
            self.table_texture.fill(TABLE_GREEN)
    
    def load_ui_elements(self, folder="assets"):
        """Load UI elements like icons and decorative graphics"""
        ui_elements = ["team_badge", "player_turn", "settings_icon", "info_panel"]
        
        for element in ui_elements:
            try:
                img = pygame.image.load(os.path.join(folder, f"{element}.png"))
                self.ui_elements[element] = img
            except:
                # Create placeholder UI elements
                self.ui_elements[element] = pygame.Surface((32, 32))
                self.ui_elements[element].fill(LIGHT_GRAY)
    
    def try_load_sounds(self, folder="assets"):
        """Try to load game sounds - gracefully fails if no audio device"""
        sound_files = {
            "card_play": "card_play.wav",
            "card_pickup": "card_pickup.wav",
            "win": "win.wav",
            "lose": "lose.wav"
        }
        
        # Try to initialize pygame mixer - skip if it fails
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.sound_enabled = True
            
            # Load sound files
            for sound_name, filename in sound_files.items():
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(os.path.join(folder, filename))
                except:
                    # Create silent sound if file not found
                    self.sounds[sound_name] = None
        except Exception as e:
            print(f"Sound disabled: {e}")
            self.sound_enabled = False
    
    def get_card_image(self, card):
        """Get the image for a specific card"""
        return self.card_images.get(card)
    
    def get_card_back(self, team_id):
        """Get the card back image for a specific team"""
        return self.card_backs.get(team_id, self.card_backs[0])
    
    def play_sound(self, sound_name):
        """Play a sound by name, safely"""
        if not self.sound_enabled:
            return
            
        sound = self.sounds.get(sound_name)
        if sound:
            try:
                sound.play()
            except:
                pass  # Silently fail if sound playback fails

class NetworkManager:
    """Manages network communication for multiplayer games over the internet"""
    def __init__(self, is_host=False, host='localhost', port=5555):
        self.is_host = is_host
        self.host = host
        self.port = port
        self.socket = None
        self.clients = []
        self.connected = False
        self.player_id = 0 if is_host else None
        self.game_state = None
        self.message_queue = deque()
        self.move_queue = deque()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.public_ip = None
        
    def start_server(self):
        """Initialize server socket for host player with improved robustness"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Permettiamo il riuso dell'indirizzo per un riavvio più veloce del server
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Binding su tutte le interfacce (0.0.0.0) per accettare connessioni esterne
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(4)  # Allow up to 4 connections
            self.connected = True
            
            # Ottieni l'IP pubblico per connessioni internet
            self.public_ip = self.get_public_ip()
            
            # Start thread to accept connections
            threading.Thread(target=self.accept_connections, daemon=True).start()
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def connect_to_server(self):
        """Connect to the host server as a client with improved connection handling"""
        self.reconnect_attempts = 0
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Impostiamo un timeout per la connessione
                self.socket.settimeout(10)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None)  # Reset timeout dopo la connessione
                self.connected = True
                
                # Start thread to receive game state updates
                threading.Thread(target=self.receive_updates, daemon=True).start()
                return True
            except Exception as e:
                self.reconnect_attempts += 1
                print(f"Tentativo {self.reconnect_attempts}/{self.max_reconnect_attempts} fallito: {e}")
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    time.sleep(2)  # Attendi prima di riprovare
        
        print("Impossibile connettersi al server dopo multipli tentativi")
        return False
    
    def accept_connections(self):
        """Accept connections from other players (for host)"""
        while len(self.clients) < 3:  # Wait for 3 more players
            try:
                client, addr = self.socket.accept()
                player_id = len(self.clients) + 1  # Assign player ID (host is 0)
                
                # Send player ID to client
                client.sendall(pickle.dumps({"type": "player_id", "id": player_id}))
                
                self.clients.append((client, player_id))
                print(f"Player {player_id} connected from {addr}")
                
                # Start thread to handle this client
                threading.Thread(target=self.handle_client, 
                                args=(client, player_id),
                                daemon=True).start()
                
                # Add message to queue
                self.message_queue.append(f"Player {player_id} connected")
                
                # If all players connected, start the game
                if len(self.clients) == 3:
                    self.message_queue.append("All players connected, starting game...")
                    self.broadcast_start_game()
            except Exception as e:
                print(f"Error accepting connection: {e}")
                break
    
    def handle_client(self, client_socket, player_id):
        """Handle communication with a specific client"""
        while self.connected:
            try:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                    
                message = pickle.loads(data)
                
                # Process different message types
                if message["type"] == "move":
                    # Add move to queue for processing
                    self.move_queue.append((player_id, message["move"]))
                elif message["type"] == "chat":
                    # Add chat message to queue
                    self.message_queue.append(f"Player {player_id}: {message['text']}")
                
            except Exception as e:
                print(f"Error handling client {player_id}: {e}")
                break
        
        # Client disconnected
        print(f"Player {player_id} disconnected")
        self.message_queue.append(f"Player {player_id} disconnected")
    
    def receive_updates(self):
        """Receive game state updates from server (for clients)"""
        while self.connected:
            try:
                data = self.socket.recv(8192)  # Larger buffer for game state
                if not data:
                    break
                    
                message = pickle.loads(data)
                
                # Process different message types
                if message["type"] == "player_id":
                    self.player_id = message["id"]
                    print(f"Assigned player ID: {self.player_id}")
                elif message["type"] == "game_state":
                    self.game_state = message["state"]
                elif message["type"] == "start_game":
                    self.message_queue.append("Game starting!")
                elif message["type"] == "chat":
                    self.message_queue.append(f"Player {message['player_id']}: {message['text']}")
                
            except Exception as e:
                print(f"Error receiving updates: {e}")
                break
        
        self.connected = False
        self.message_queue.append("Disconnected from server")
    
    def send_move(self, move):
        """Send a move to the server (for clients) or broadcast it (for host)"""
        message = {"type": "move", "move": move}
        
        if self.is_host:
            # Process move locally and broadcast updated state
            self.move_queue.append((0, move))  # Host is player 0
        else:
            # Send move to server
            try:
                self.socket.sendall(pickle.dumps(message))
            except Exception as e:
                print(f"Error sending move: {e}")
                self.connected = False
    
    def send_chat(self, text):
        """Send a chat message"""
        message = {"type": "chat", "text": text}
        
        if self.is_host:
            # Add to local message queue and broadcast
            self.message_queue.append(f"Player 0: {text}")
            self.broadcast_chat(0, text)
        else:
            # Send to server
            try:
                self.socket.sendall(pickle.dumps(message))
            except Exception as e:
                print(f"Error sending chat: {e}")
                self.connected = False
    
    def broadcast_game_state(self):
        """Broadcast current game state to all clients (host only)"""
        if not self.is_host:
            return
            
        message = {"type": "game_state", "state": self.game_state}
        data = pickle.dumps(message)
        
        for client, _ in self.clients:
            try:
                client.sendall(data)
            except:
                pass  # Client probably disconnected
    
    def broadcast_start_game(self):
        """Broadcast game start signal to all clients (host only)"""
        if not self.is_host:
            return
            
        message = {"type": "start_game"}
        data = pickle.dumps(message)
        
        for client, _ in self.clients:
            try:
                client.sendall(data)
            except:
                pass
    
    def broadcast_chat(self, player_id, text):
        """Broadcast chat message to all clients (host only)"""
        if not self.is_host:
            return
            
        message = {"type": "chat", "player_id": player_id, "text": text}
        data = pickle.dumps(message)
        
        for client, _ in self.clients:
            try:
                client.sendall(data)
            except:
                pass
    
    def close(self):
        """Close network connection"""
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            
        self.socket = None
        self.clients = []

    def get_public_ip(self):
        """Get public IP address for internet play"""
        try:
            import urllib.request
            # Usiamo un servizio che restituisce solo l'IP come testo puro
            response = urllib.request.urlopen('https://api.ipify.org')
            self.public_ip = response.read().decode('utf8')
            return self.public_ip
        except Exception as e:
            print(f"Impossibile recuperare l'IP pubblico: {e}")
            # Fallback su IP locale se non possiamo ottenere quello pubblico
            return get_local_ip()

class BaseScreen:
    """Base class for all game screens"""
    def __init__(self, app):
        self.app = app
        self.done = False
        self.next_screen = None
    
    def handle_events(self, events):
        """Handle pygame events"""
        pass
    
    def update(self):
        """Update screen state"""
        pass
    
    def draw(self, surface):
        """Draw screen content"""
        pass
    
    def enter(self):
        """Called when entering this screen"""
        self.done = False
    
    def exit(self):
        """Called when exiting this screen"""
        pass

class GameModeScreen(BaseScreen):
    """Screen for selecting game mode"""
    def __init__(self, app):
        super().__init__(app)
        
        # Aggiungi stato di caricamento
        self.loading = False
        self.loading_animation = None
        self.loading_start_time = 0
        self.loading_message = ""
        
        # Background image
        self.bg_image = None  # Will be set in enter()
        
        # UI Elements
        center_x = SCREEN_WIDTH // 2
        self.title_font = pygame.font.SysFont(None, 72)
        self.info_font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)  # Add this line to define small_font
        
        # Buttons
        button_width = 400
        button_height = 60
        button_spacing = 20
        button_start_y = 200
        
        self.buttons = [
            Button(center_x - button_width // 2, 
                  button_start_y, 
                  button_width, button_height,
                  "Single Player (vs 3 AI)",
                  DARK_BLUE, WHITE),
                  
            Button(center_x - button_width // 2, 
                  button_start_y + button_height + button_spacing, 
                  button_width, button_height,
                  "2 Players (Team vs AI)",
                  DARK_BLUE, WHITE),
                  
            Button(center_x - button_width // 2, 
                  button_start_y + 2 * (button_height + button_spacing), 
                  button_width, button_height,
                  "4 Players (Human vs Human)",
                  DARK_BLUE, WHITE),
                  
            Button(center_x - button_width // 2, 
                  button_start_y + 3 * (button_height + button_spacing), 
                  button_width, button_height,
                  "Host Online Game",
                  DARK_BLUE, WHITE),
                  
            Button(center_x - button_width // 2, 
                  button_start_y + 4 * (button_height + button_spacing), 
                  button_width, button_height,
                  "Join Online Game",
                  DARK_BLUE, WHITE),
        ]
        
        # Difficulty selection
        self.difficulty_buttons = [
            Button(center_x - 250, button_start_y + 5 * (button_height + button_spacing), 
                  150, 40, "Easy", DARK_GREEN, WHITE),
            Button(center_x - 75, button_start_y + 5 * (button_height + button_spacing), 
                  150, 40, "Medium", DARK_BLUE, WHITE),
            Button(center_x + 100, button_start_y + 5 * (button_height + button_spacing), 
                  150, 40, "Hard", HIGHLIGHT_RED, WHITE),
        ]
        self.selected_difficulty = 2  # Default: Hard
        
        # IP input for joining online games
        self.ip_input = ""
        self.ip_input_active = False
        self.ip_input_rect = pygame.Rect(center_x - 225,  # Centered based on new width
                                    button_start_y + 6 * (button_height + button_spacing), 
                                    450, 40)  # Wider input field
        
        # Status message
        self.status_message = ""
        
        self.host_screen_active = False
        self.online_mode_buttons = []
        self.selected_online_mode = 0  # 0: 4 Players, 1: 2v2 with AI
    
    def enter(self):
        super().enter()
        self.bg_image = self.app.resources.background  # Aggiorna l'immagine
        self.setup_layout()  # Set up initial layout
    
    def setup_layout(self):
        """Set up responsive layout for mode selection screen"""
        # Aggiorna il background con l'immagine ridimensionata
        self.bg_image = self.app.resources.background
        # Get current window dimensions
        width = self.app.window_width
        height = self.app.window_height
        
        # Update font sizes based on window dimensions
        self.title_font = pygame.font.SysFont(None, int(height * 0.09))  # ~9% of height
        self.info_font = pygame.font.SysFont(None, int(height * 0.03))   # ~3% of height
        self.small_font = pygame.font.SysFont(None, int(height * 0.023))  # Add this line for small_font
        
        # Button dimensions and positioning
        center_x = width // 2
        button_width = int(width * 0.4)
        button_height = int(height * 0.08)
        button_spacing = int(height * 0.03)
        button_start_y = int(height * 0.25)
        
        # Main game mode buttons
        self.buttons = [
            Button(center_x - button_width // 2, 
                button_start_y, 
                button_width, button_height,
                "Single Player (vs 3 AI)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2, 
                button_start_y + button_height + button_spacing, 
                button_width, button_height,
                "2 Players (Team vs AI)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2, 
                button_start_y + 2 * (button_height + button_spacing), 
                button_width, button_height,
                "4 Players (Human vs Human)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2, 
                button_start_y + 3 * (button_height + button_spacing), 
                button_width, button_height,
                "Host Online Game",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2, 
                button_start_y + 4 * (button_height + button_spacing), 
                button_width, button_height,
                "Join Online Game",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
        ]
        
        # Difficulty selection buttons
        diff_button_width = int(width * 0.15)
        diff_button_height = int(height * 0.05)
        
        self.difficulty_buttons = [
            Button(center_x - int(width * 0.25), button_start_y + 5 * (button_height + button_spacing), 
                diff_button_width, diff_button_height, "Easy", DARK_GREEN, WHITE, 
                font_size=int(height * 0.025)),
            Button(center_x - diff_button_width//2, button_start_y + 5 * (button_height + button_spacing), 
                diff_button_width, diff_button_height, "Medium", DARK_BLUE, WHITE, 
                font_size=int(height * 0.025)),
            Button(center_x + int(width * 0.1), button_start_y + 5 * (button_height + button_spacing), 
                diff_button_width, diff_button_height, "Hard", HIGHLIGHT_RED, WHITE, 
                font_size=int(height * 0.025)),
        ]
        
        # IP input for joining online games
        self.ip_input_rect = pygame.Rect(center_x - int(width * 0.225),  # Centered based on new width
                                    button_start_y + 6 * (button_height + button_spacing),
                                    int(width * 0.45), int(height * 0.05))  # Wider input field
        
        # NUOVA PARTE PER EVITARE SOVRAPPOSIZIONI
        # Raccogliamo tutti gli elementi UI
        ui_elements = []
        
        # Aggiungi tutti i bottoni principali (priorità 1)
        for button in self.buttons:
            ui_elements.append((button.rect, 1))
        
        # Aggiungi i bottoni di difficoltà (priorità 2)
        for button in self.difficulty_buttons:
            ui_elements.append((button.rect, 2))
        
        # Aggiungi la casella di input IP (priorità 3)
        ui_elements.append((self.ip_input_rect, 3))
        
        # Usa il LayoutManager per riposizionare gli elementi evitando sovrapposizioni
        from layout import LayoutManager
        
        # Riposiziona gli elementi mantenendo l'allineamento orizzontale dei bottoni principali
        # ma permettendo spostamenti verticali per evitare sovrapposizioni
        positioned_rects = []
        
        # Posiziona prima i bottoni principali in modo ordinato (mantieni x, ma aggiusta y se necessario)
        for i, (rect, _) in enumerate([item for item in ui_elements if item[1] == 1]):
            if i == 0:  # Il primo bottone mantiene la sua posizione originale
                positioned_rects.append(rect)
                continue
                
            # Per gli altri bottoni, verifica che non ci siano sovrapposizioni con i precedenti
            test_rect = rect.copy()
            
            # Incrementa y finché non si trova una posizione senza sovrapposizioni
            for distance in range(0, 500, 5):
                test_rect.y = rect.y + distance
                padded_rect = test_rect.inflate(5, 5)  # Aggiungi un piccolo padding
                
                if not any(LayoutManager.check_collision(padded_rect, pos_rect) for pos_rect in positioned_rects):
                    rect.y = test_rect.y
                    break
                    
            positioned_rects.append(rect)
        
        # Posiziona i bottoni di difficoltà e la casella di input IP
        remaining_elements = [(rect, prio) for rect, prio in ui_elements if prio > 1]
        new_rects = LayoutManager.arrange_elements(remaining_elements, width, height)
        
        # Aggiorna le posizioni dei rettangoli originali
        rect_index = 0
        for i, (rect, prio) in enumerate(ui_elements):
            if prio > 1:  # Bottoni di difficoltà e input IP
                rect.x = new_rects[rect_index].x
                rect.y = new_rects[rect_index].y
                rect_index += 1
    
    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.done = True
                self.next_screen = None
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check buttons
                pos = pygame.mouse.get_pos()
                
                # Se siamo nella schermata di scelta della modalità online
                if self.host_screen_active:
                    # Controllo clic sui pulsanti delle modalità online
                    for i, button in enumerate(self.online_mode_buttons):
                        if button.is_clicked(pos):
                            if i == 0 or i == 1:  # 4 Players o 2v2 with AI
                                self.selected_online_mode = i
                                # Ora non avviamo immediatamente il gioco in nessun caso,
                                # aspettiamo che l'utente prema il pulsante Start
                            elif i == 2:  # Back
                                self.host_screen_active = False
                    
                    # Controllo clic sui pulsanti di difficoltà (solo per modalità Team vs AI)
                    if self.selected_online_mode == 1 and hasattr(self, 'online_difficulty_buttons'):
                        for i, btn in enumerate(self.online_difficulty_buttons):
                            if btn.is_clicked(pos):
                                self.selected_difficulty = i
                    
                    # Controllo clic sul pulsante Start
                    if hasattr(self, 'start_button') and self.start_button.is_clicked(pos):
                        self.host_online_game()
                    
                    return
                
                # Resto del codice per la schermata principale (invariato)
                # Check if IP input box clicked
                if self.ip_input_rect.collidepoint(pos):
                    self.ip_input_active = True
                else:
                    self.ip_input_active = False
                
                # Check if any main button clicked
                for i, button in enumerate(self.buttons):
                    if button.is_clicked(pos):
                        self.handle_button_click(i)
                
                # Check if any difficulty button clicked
                for i, button in enumerate(self.difficulty_buttons):
                    if button.is_clicked(pos):
                        self.selected_difficulty = i
            
            elif event.type == pygame.KEYDOWN and self.ip_input_active:
                if event.key == pygame.K_RETURN:
                    # Join game with current IP
                    self.join_online_game()
                elif event.key == pygame.K_BACKSPACE:
                    self.ip_input = self.ip_input[:-1]
                else:
                    if len(self.ip_input) < 25:  # Increased limit for longer IP addresses or hostnames
                        self.ip_input += event.unicode
    

    def handle_button_click(self, button_index):
        """Handle button clicks"""
        if button_index == 0:
            # Single Player - Mostra animazione di caricamento
            self.loading = True
            self.loading_animation = LoadingAnimation(self.app.window_width, self.app.window_height)
            self.loading_start_time = pygame.time.get_ticks()
            self.loading_message = "Caricamento bot AI in corso"
            
            # Configura il gioco ma non passare direttamente alla prossima schermata
            self.app.game_config = {
                "mode": "single_player",
                "human_players": 1,
                "ai_players": 3,
                "difficulty": self.selected_difficulty
            }
            return
            
        elif button_index == 1:
            # 2 Players (Team) - Mostra animazione di caricamento
            self.loading = True
            self.loading_animation = LoadingAnimation(self.app.window_width, self.app.window_height)
            self.loading_start_time = pygame.time.get_ticks()
            self.loading_message = "Caricamento bot AI in corso"
            
            # Configura il gioco ma non passare direttamente alla prossima schermata
            self.app.game_config = {
                "mode": "team_vs_ai",
                "human_players": 2,
                "ai_players": 2,
                "difficulty": self.selected_difficulty
            }
            return
            
        # Solo per le modalità senza bot, passa direttamente alla prossima schermata
        elif button_index == 2:
            # 4 Players (Local)
            self.done = True
            self.next_screen = "game"
            self.app.game_config = {
                "mode": "local_multiplayer",
                "human_players": 4,
                "ai_players": 0
            }
        elif button_index == 3:
            # Host Online Game - ora mostra schermata di scelta
            self.host_screen_active = True
            self.setup_online_choice_buttons()
        elif button_index == 4:
            # Join Online Game
            if self.ip_input:
                self.join_online_game()
            else:
                self.status_message = "Please enter an IP address"
                # Make the IP input active to focus it after showing the error
                self.ip_input_active = True
    
    def host_online_game(self):
        """Host an online game with internet support"""
        self.app.network = NetworkManager(is_host=True)
        if self.app.network.start_server():
            # Ottieni sia l'IP locale che quello pubblico
            local_ip = get_local_ip()
            public_ip = self.app.network.public_ip
            
            # Mostra entrambi gli IP nello status message
            self.status_message = f"Server attivo! LAN: {local_ip} | Internet: {public_ip} | Porta: 5555"
            
            # Aggiungi queste informazioni alla message queue in modo organizzato
            if self.app.network:
                self.app.network.message_queue.append("Server avviato con successo!")
                self.app.network.message_queue.append(f"IP Locale (LAN): {local_ip}")
                self.app.network.message_queue.append(f"IP Pubblico (Internet): {public_ip}")
                self.app.network.message_queue.append("Per LAN: usa l'IP Locale")
                self.app.network.message_queue.append("Per Internet: usa l'IP Pubblico + port forwarding (porta 5555)")

            if self.selected_online_mode == 0:
                # 4 Players (All Human)
                self.done = True
                self.next_screen = "game"
                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": True,
                    "player_id": 0,
                    "online_type": "all_human"
                }
            else:
                # 2 vs 2 (Team vs AI) - Mostra animazione di caricamento
                self.loading = True
                self.loading_animation = LoadingAnimation(self.app.window_width, self.app.window_height)
                self.loading_start_time = pygame.time.get_ticks()
                self.loading_message = "Caricamento bot AI in corso"
                
                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": True,
                    "player_id": 0,
                    "online_type": "team_vs_ai",
                    "difficulty": self.selected_difficulty
                }
        else:
            self.status_message = "Failed to start server"
    
    def join_online_game(self):
        """Join an online game"""
        host = self.ip_input if self.ip_input else "localhost"
        self.app.network = NetworkManager(is_host=False, host=host)
        
        if self.app.network.connect_to_server():
            self.done = True
            self.next_screen = "game"
            self.app.game_config = {
                "mode": "online_multiplayer",
                "is_host": False,
                "player_id": None  # Will be set by server
            }
        else:
            self.status_message = f"Failed to connect to {host}"
    
    def draw(self, surface):
        # Draw background
        surface.blit(self.bg_image, (0, 0))
        
        # Get current window dimensions
        width = self.app.window_width
        height = self.app.window_height
        center_x = width // 2
        
        # Se siamo nella schermata di scelta della modalità online
        if self.host_screen_active:
            # Draw title
            title_surf = self.title_font.render("Choose Online Mode", True, GOLD)
            title_rect = title_surf.get_rect(center=(center_x, height * 0.15))
            surface.blit(title_surf, title_rect)
            
            # Draw buttons
            for i, button in enumerate(self.online_mode_buttons):
                # Highlight selected button for game modes
                if i < 2 and i == self.selected_online_mode:
                    pygame.draw.rect(surface, GOLD, button.rect.inflate(6, 6), 3, border_radius=8)
                button.draw(surface)
            
            # Draw instructions
            if self.selected_online_mode == 0:
                info_text = "You'll need 3 more human players to join"
            else:
                info_text = "You'll need 1 more human player to join (for your team)"
            
            info_surf = self.info_font.render(info_text, True, WHITE)
            info_rect = info_surf.get_rect(center=(center_x, height - height * 0.15))
            surface.blit(info_surf, info_rect)
            
            # Draw difficulty selection if we're doing Team vs AI
            if self.selected_online_mode == 1:
                difficulty_y = height - height * 0.25
                difficulty_label = self.info_font.render("AI Difficulty:", True, WHITE)
                label_rect = difficulty_label.get_rect(
                    center=(center_x, difficulty_y)
                )
                surface.blit(difficulty_label, label_rect)
                
                # Ridisegna i pulsanti di difficoltà in una posizione diversa
                diff_button_width = int(width * 0.12)
                diff_button_height = int(height * 0.05)
                diff_spacing = int(width * 0.02)
                
                diff_buttons_total_width = 3 * diff_button_width + 2 * diff_spacing
                diff_start_x = center_x - diff_buttons_total_width // 2
                
                self.online_difficulty_buttons = []  # Resetta la lista
                
                for i, label in enumerate(["Easy", "Medium", "Hard"]):
                    color = DARK_GREEN if i == 0 else DARK_BLUE if i == 1 else HIGHLIGHT_RED
                    btn = Button(
                        diff_start_x + i * (diff_button_width + diff_spacing),
                        difficulty_y + label_rect.height + 10,
                        diff_button_width, diff_button_height,
                        label, color, WHITE,
                        font_size=int(height * 0.025)
                    )
                    
                    self.online_difficulty_buttons.append(btn)
                    
                    # Highlight selected difficulty
                    if i == self.selected_difficulty:
                        pygame.draw.rect(surface, GOLD, btn.rect.inflate(6, 6), 3, border_radius=8)
                    btn.draw(surface)
            
            # Disegna il pulsante Start per ENTRAMBE le modalità
            # Ma posizionalo più in basso per la modalità Team vs AI per non sovrapporsi ai pulsanti di difficoltà
            start_button_width = int(width * 0.25)
            start_button_height = int(height * 0.08)
            
            start_text = "Start Game"
            # Aggiusta la posizione e il testo del pulsante in base alla modalità
            if self.selected_online_mode == 0:  # 4 Players (All Human)
                start_y = height - int(height * 0.1)
            else:  # 2 vs 2 (Team vs AI)
                start_y = height - int(height * 0.1)
                
            self.start_button = Button(
                center_x - start_button_width // 2,
                start_y,
                start_button_width, start_button_height,
                start_text,
                DARK_GREEN, WHITE, font_size=int(height * 0.03)
            )
            
            self.start_button.draw(surface)
            
            return
        
        # CODICE DELLA SCHERMATA PRINCIPALE:
        
        # Draw title
        title_surf = self.title_font.render("Scopone a Coppie", True, GOLD)
        title_rect = title_surf.get_rect(center=(center_x, height * 0.1))
        surface.blit(title_surf, title_rect)
        
        # Draw buttons
        for button in self.buttons:
            button.draw(surface)
        
        # Draw difficulty buttons
        difficulty_y = self.difficulty_buttons[0].rect.centery
        difficulty_label = self.info_font.render("AI Difficulty:", True, WHITE)
        label_rect = difficulty_label.get_rect(
            right=self.difficulty_buttons[0].rect.left - width * 0.01,
            centery=difficulty_y
        )
        surface.blit(difficulty_label, label_rect)
        
        for i, button in enumerate(self.difficulty_buttons):
            # Highlight selected difficulty
            if i == self.selected_difficulty:
                pygame.draw.rect(surface, GOLD, button.rect.inflate(6, 6), 3, border_radius=8)
            button.draw(surface)
        
        # Draw IP input box
        ip_color = LIGHT_BLUE if self.ip_input_active else DARK_BLUE
        pygame.draw.rect(surface, ip_color, self.ip_input_rect, border_radius=5)
        pygame.draw.rect(surface, WHITE, self.ip_input_rect, 2, border_radius=5)

        # Draw IP input text - MODIFIED PART
        if self.status_message and "Please enter an IP address" in self.status_message:
            # When there's an error about IP address, show nothing in the input box
            ip_text = ""
            # We'll let the red status message communicate the error
        else:
            # Normal display behavior
            ip_text = self.ip_input if self.ip_input else "Enter IP to join"

        ip_surf = self.info_font.render(ip_text, True, WHITE)
        ip_rect = ip_surf.get_rect(center=self.ip_input_rect.center)
        surface.blit(ip_surf, ip_rect)
        
        # Aggiungi istruzioni per il gioco online nella schermata di join
        if self.ip_input_active:
            help_text = [
                "Per giocare su internet:", 
                "1. L'host deve configurare port forwarding sul router (porta 5555)",
                "2. Inserisci l'IP pubblico dell'host (non l'IP locale)",
                "3. Assicurati che non ci siano firewall che bloccano la porta 5555"
            ]
            
            help_y = self.ip_input_rect.bottom + 15
            for line in help_text:
                help_surf = self.small_font.render(line, True, LIGHT_GRAY)
                help_rect = help_surf.get_rect(left=self.ip_input_rect.left, top=help_y)
                surface.blit(help_surf, help_rect)
                help_y += 20
        
        # Draw status message if any
        if self.status_message:
            status_surf = self.info_font.render(self.status_message, True, HIGHLIGHT_RED)
            status_rect = status_surf.get_rect(center=(center_x, height - height * 0.07))
            surface.blit(status_surf, status_rect)
        
        # Draw info text
        info_text = "Select a game mode to begin"
        info_surf = self.info_font.render(info_text, True, WHITE)
        info_rect = info_surf.get_rect(center=(center_x, height - height * 0.03))
        surface.blit(info_surf, info_rect)
        
        # Se siamo in stato di caricamento, disegna l'animazione sopra tutto
        if self.loading:
            self.loading_animation.draw(surface, self.loading_message)
        
    def setup_online_choice_buttons(self):
        """Configura i pulsanti per la scelta della modalità online"""
        width = self.app.window_width
        height = self.app.window_height
        center_x = width // 2
        
        button_width = int(width * 0.4)
        button_height = int(height * 0.08)
        button_spacing = int(height * 0.05)
        
        title_y = int(height * 0.25)
        button_start_y = title_y + int(height * 0.15)
        
        self.online_mode_buttons = [
            Button(center_x - button_width // 2,
                button_start_y,
                button_width, button_height,
                "4 Players (All Human)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2,
                button_start_y + button_height + button_spacing,
                button_width, button_height,
                "2 vs 2 (Team vs AI)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2,
                button_start_y + 2 * (button_height + button_spacing),
                button_width, button_height,
                "Back",
                HIGHLIGHT_RED, WHITE, font_size=int(height * 0.03))
        ]
        
        # Aggiungiamo un pulsante Start separato per la modalità Team vs AI
        start_button_width = int(width * 0.25)
        self.start_button = Button(
            center_x - start_button_width // 2,
            height - int(height * 0.1),
            start_button_width, button_height,
            "Start Game",
            DARK_GREEN, WHITE, font_size=int(height * 0.03)
        )
        
        # Inizializza la lista per i pulsanti di difficoltà
        self.online_difficulty_buttons = []
        
        
    def update(self):
        """Update screen state"""
        if self.loading:
            # Se siamo in stato di caricamento, aggiorna l'animazione
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.loading_start_time
            
            # Aggiorna l'animazione
            self.loading_animation.update()
            
            # Dopo 2 secondi, completa il caricamento e vai alla schermata di gioco
            if elapsed > 2000:  # 2 secondi di animazione
                self.loading = False
                self.done = True
                self.next_screen = "game"

class PlayerInfo:
    """Stores display information about a player"""
    def __init__(self, player_id, name=None, team_id=None, is_human=True, is_ai=False):
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.team_id = team_id or (0 if player_id in [0, 2] else 1)
        self.is_human = is_human
        self.is_ai = is_ai
        self.hand_cards = []
        self.hand_rect = pygame.Rect(0, 0, 100, 100)  # Will be set properly in layout
        self.avatar_rect = pygame.Rect(0, 0, 50, 50)
        self.info_rect = pygame.Rect(0, 0, 100, 50)
        
    def set_hand(self, cards):
        self.hand_cards = cards.copy() if cards else []

class GameScreen(BaseScreen):
    """Main game screen for playing Scopone"""
    def __init__(self, app):
        super().__init__(app)
        
        # Game state
        self.env = None
        self.players = []
        self.current_player_id = 0
        self.local_player_id = 0
        self.selected_hand_card = None
        self.selected_table_cards = set()
        self.animations = []
        self.game_over = False
        self.final_breakdown = None
        self.waiting_for_other_player = False
        self.ai_thinking = False
        self.ai_move_timer = 0
        self.messages = []
        self.message_scroll_offset = 0  # Offset per lo scrolling dei messaggi
        self.max_visible_messages = 5   # Numero massimo di messaggi visibili contemporaneamente
        
        # AI players
        self.ai_controllers = {}
        self.ai_difficulty = 1  # Default: Medium
        
        # UI elements
        self.table_rect = pygame.Rect(SCREEN_WIDTH // 2 - 375, 
                                      SCREEN_HEIGHT // 2 - 225, 
                                      750, 450)
        
        # Buttons
        self.confirm_button = Button(SCREEN_WIDTH - 160, SCREEN_HEIGHT - 80, 
                                    140, 50, "Play Card", 
                                    (0, 150, 0), WHITE)
        
        self.new_game_button = Button(20, 20, 140, 50, "New Game", 
                                     DARK_BLUE, WHITE)
        
        # Message log
        self.message_log_rect = pygame.Rect(20, SCREEN_HEIGHT - 150, 
                                           300, 130)
        
        # Fonts
        self.title_font = pygame.font.SysFont(None, 32)
        self.info_font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        
        # Status variables
        self.message_timeout = 0
        self.status_message = ""
        
        # Game over button (for click detection)
        self.game_over_button_rect = None
    
    def enter(self):
        super().enter()
        
        # Initialize game based on config
        self.initialize_game()
        
        # Set up player info
        self.setup_players()
        
        # Set up layout
        self.setup_layout()
        
        # Clear card angles and other state
        self.game_over_button_rect = None
    
    def initialize_game(self):
        """Initialize game environment and state with random starting player"""
        config = self.app.game_config
        mode = config.get("mode", "single_player")
        
        # Set difficulty
        self.ai_difficulty = config.get("difficulty", 2)  # Default: Hard
        
        # Create game environment
        self.env = ScoponeEnvMA()
        
        # Imposta un giocatore iniziale casuale (0, 1, 2 o 3)
        random_starter = random.randint(0, 3)
        
        # Resetta l'ambiente con il giocatore iniziale casuale
        self.env.reset(starting_player=random_starter)
        
        # Set up AI controllers if needed
        if mode in ["single_player", "team_vs_ai"]:
            self.setup_ai_controllers()
        
        # Set up player ID (for online games)
        if mode == "online_multiplayer":
            self.local_player_id = config.get("player_id", 0)
    
    def setup_ai_controllers(self):
        """Set up AI controllers based on game mode"""
        config = self.app.game_config
        mode = config.get("mode", "single_player")
        
        if mode == "single_player":
            # Player 0 is human, rest are AI
            for player_id in [1, 2, 3]:
                team_id = 0 if player_id in [0, 2] else 1
                self.ai_controllers[player_id] = DQNAgent(team_id=team_id)
                
                # Load checkpoint if available
                checkpoint_path = f"scopone_checkpoint_team{team_id}.pth"
                if os.path.exists(checkpoint_path):
                    self.ai_controllers[player_id].load_checkpoint(checkpoint_path)
                
                # Set exploration rate based on difficulty
                # Easy: More random actions, Hard: More optimal actions
                if self.ai_difficulty == 0:  # Easy
                    self.ai_controllers[player_id].epsilon = 0.3
                elif self.ai_difficulty == 1:  # Medium
                    self.ai_controllers[player_id].epsilon = 0.1
                else:  # Hard
                    self.ai_controllers[player_id].epsilon = 0.01
        
        elif mode == "team_vs_ai":
            # Players 0 and 2 are human team, 1 and 3 are AI team
            for player_id in [1, 3]:
                self.ai_controllers[player_id] = DQNAgent(team_id=1)
                
                # Load checkpoint if available
                checkpoint_path = "scopone_checkpoint_team1.pth"
                if os.path.exists(checkpoint_path):
                    self.ai_controllers[player_id].load_checkpoint(checkpoint_path)
                
                # Set exploration rate based on difficulty
                if self.ai_difficulty == 0:  # Easy
                    self.ai_controllers[player_id].epsilon = 0.3
                elif self.ai_difficulty == 1:  # Medium
                    self.ai_controllers[player_id].epsilon = 0.1
                else:  # Hard
                    self.ai_controllers[player_id].epsilon = 0.01
    
    def setup_players(self):
        """Set up player info objects with improved perspective handling"""
        config = self.app.game_config
        mode = config.get("mode", "single_player")
        
        self.players = []
        
        for player_id in range(4):
            team_id = 0 if player_id in [0, 2] else 1
            
            if mode == "single_player":
                is_human = (player_id == 0)
                is_ai = not is_human
                name = "You" if is_human else f"AI {player_id}"
            elif mode == "team_vs_ai":
                is_human = player_id in [0, 2]
                is_ai = not is_human
                if is_human:
                    name = "You" if player_id == 0 else "Partner"
                else:
                    name = f"AI {player_id}"
            elif mode == "local_multiplayer":
                is_human = True
                is_ai = False
                name = f"Player {player_id}"
            elif mode == "online_multiplayer":
                is_human = (player_id == self.local_player_id)
                is_ai = False
                name = "You" if is_human else f"Player {player_id}"
            
            player = PlayerInfo(
                player_id=player_id,
                name=name,
                team_id=team_id,
                is_human=is_human,
                is_ai=is_ai
            )
            
            self.players.append(player)
        
        # For online multiplayer - set up player perspective mapping
        if mode == "online_multiplayer":
            self.setup_player_perspective()

    def setup_player_perspective(self):
        """Set up visualization for different client perspectives
        So each client sees their hand at the bottom (position 0)"""
        if not hasattr(self, 'visual_positions'):
            self.visual_positions = [0, 1, 2, 3]  # Default mapping
        
        # Check if local_player_id is None (not yet set by server)
        if self.local_player_id is None:
            print("Warning: local_player_id is None, using default perspective mapping")
            return  # Keep default mapping until player ID is assigned
        
        # Calculate the rotation needed based on local player ID
        # If player is player_id 0: no change
        # If player is player_id 1: rotate 90 degrees counterclockwise (player sees themselves at bottom)
        # If player is player_id 2: rotate 180 degrees (player sees themselves at bottom)
        # If player is player_id 3: rotate 90 degrees clockwise (player sees themselves at bottom)
        
        # Calculate shifted positions
        rotation_steps = self.local_player_id
        self.visual_positions = [(i - rotation_steps) % 4 for i in range(4)]
        
        # Print visual positions mapping for debugging
        print(f"Visual positions mapping: {self.visual_positions}")
        print(f"Local player {self.local_player_id} is at visual position {self.visual_positions[self.local_player_id]}")

    def get_visual_position(self, logical_player_id):
        """Convert logical player ID to visual position based on client perspective"""
        if not hasattr(self, 'visual_positions') or self.local_player_id is None:
            return logical_player_id  # Use default mapping if no visual positions set yet
        return self.visual_positions[logical_player_id]
    
    def setup_layout(self):
        """Set up the screen layout with proper perspective handling for network play"""
        # Get current window dimensions
        width = self.app.window_width
        height = self.app.window_height
        
        # Check if players list is populated before accessing it
        if not hasattr(self, 'players') or len(self.players) < 4:
            return
        
        # Calculate card dimensions based on window size
        card_width = int(width * 0.078)  # ~8% of window width
        card_height = int(card_width * 1.5)  # Maintain aspect ratio
        
        # Store card dimensions as instance variables for use in other methods
        self.card_width = card_width
        self.card_height = card_height
        
        # Calculate the spacing and layout based on current window dimensions
        card_spread = card_width * 0.7  # Cards will overlap
        max_cards = 10
        hand_width = card_width + (max_cards - 1) * card_spread
        
        # Center table area
        table_width = width * 0.7
        table_height = height * 0.5
        self.table_rect = pygame.Rect(
            width // 2 - table_width // 2,
            height // 2 - table_height // 2,
            table_width,
            table_height
        )
        
        # Player hand positions, adjusting for network perspective
        hand_positions = [
            # Bottom player
            pygame.Rect(
                width // 2 - hand_width // 2,
                height - card_height - height * 0.05,  # 5% from bottom
                hand_width,
                card_height
            ),
            # Left player
            pygame.Rect(
                width * 0.02,  # 2% from left
                height // 2 - hand_width // 2,
                card_height,  # Swapped for vertical layout
                hand_width
            ),
            # Top player
            pygame.Rect(
                width // 2 - hand_width // 2,
                height * 0.05,  # 5% from top
                hand_width,
                card_height
            ),
            # Right player
            pygame.Rect(
                width - card_height - width * 0.02,  # 2% from right
                height // 2 - hand_width // 2,
                card_height,  # Swapped for vertical layout
                hand_width
            )
        ]
        
        # Assign hand rectangles based on visual position
        for player in self.players:
            visual_pos = self.get_visual_position(player.player_id)
            player.hand_rect = hand_positions[visual_pos]
            
            # For debugging
            if player.player_id == self.local_player_id:
                print(f"Player {player.player_id} (You) is at visual position {visual_pos}")
        
        # Avatar positions relative to hands
        avatar_size = int(height * 0.08)
        
        for player in self.players:
            visual_pos = self.get_visual_position(player.player_id)
            
            if visual_pos == 0:  # Bottom player - avatar to the left
                player.avatar_rect = pygame.Rect(
                    max(width * 0.01, player.hand_rect.left - avatar_size - width * 0.07),
                    player.hand_rect.centery - avatar_size // 2,
                    avatar_size, avatar_size
                )
            elif visual_pos == 1:  # Left player - avatar to the right
                player.avatar_rect = pygame.Rect(
                    player.hand_rect.right + width * 0.02,
                    player.hand_rect.centery - avatar_size // 2,
                    avatar_size, avatar_size
                )
            elif visual_pos == 2:  # Top player - avatar to the left
                player.avatar_rect = pygame.Rect(
                    player.hand_rect.left - avatar_size - width * 0.02,
                    player.hand_rect.centery - avatar_size // 2,
                    avatar_size, avatar_size
                )
            elif visual_pos == 3:  # Right player - avatar to the left
                player.avatar_rect = pygame.Rect(
                    player.hand_rect.left - avatar_size - width * 0.02,
                    player.hand_rect.centery - avatar_size // 2,
                    avatar_size, avatar_size
                )
                
            # Set info rect as the same as avatar rect for compatibility
            player.info_rect = player.avatar_rect.copy()
        
        # UI elements and other layout elements - similar to original
        button_width = int(width * 0.14)
        button_height = int(height * 0.06)
        
        # New Game button - top-left corner
        self.new_game_button = Button(
            width * 0.01,  # Far left
            height * 0.01,  # Far top
            button_width, 
            button_height,
            "New Game", 
            DARK_BLUE, WHITE
        )
        
        # Play Card button - bottom-right corner
        self.confirm_button = Button(
            width - button_width - width * 0.01,
            height - button_height - height * 0.01,
            button_width, 
            button_height,
            "Play Card", 
            (0, 150, 0), WHITE
        )
        
        # Message log - top right corner
        self.message_log_rect = pygame.Rect(
            width - width * 0.25 - width * 0.02,
            height * 0.05,
            width * 0.25, 
            height * 0.18
        )
        
        # Update fonts based on screen dimensions
        self.title_font = pygame.font.SysFont(None, int(height * 0.042))
        self.info_font = pygame.font.SysFont(None, int(height * 0.031))
        self.small_font = pygame.font.SysFont(None, int(height * 0.023))
        
        # Update global card size constants
        global CARD_WIDTH, CARD_HEIGHT
        CARD_WIDTH = card_width
        CARD_HEIGHT = card_height
        
        # Resize card images
        self.app.resources.rescale_card_images(card_width, card_height)
    
    def update_player_hands(self):
        """Update player hand information from game state"""
        if not self.env:
            return
            
        gs = self.env.game_state
        
        for player in self.players:
            if player.player_id in gs["hands"]:
                player.set_hand(gs["hands"][player.player_id])
            else:
                player.set_hand([])
    
    def handle_events(self, events):
        """Handle pygame events with improved card selection logic"""
        for event in events:
            if event.type == pygame.QUIT:
                self.done = True
                self.next_screen = None
                pygame.quit()
                sys.exit()
            
            # Always check for the "New Game" button
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if self.new_game_button.is_clicked(pos):
                    self.done = True
                    self.next_screen = "mode"
                    return
            
            # Ignore input during specific game states
            if self.game_over or self.waiting_for_other_player or self.ai_thinking:
                if self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if new game button is clicked in game over screen
                    mouse_pos = pygame.mouse.get_pos()
                    if hasattr(self, 'game_over_button_rect') and self.game_over_button_rect and self.game_over_button_rect.collidepoint(mouse_pos):
                        self.done = True
                        self.next_screen = "mode"
                continue
            
            # Check if current player is controllable
            if not self.is_current_player_controllable():
                continue
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                # Process confirm button
                if self.confirm_button.is_clicked(pos):
                    self.try_make_move()
                
                # Check hand cards with visual perspective aware logic
                hand_card = self.get_card_at_position(pos, area="hand")
                if hand_card:
                    if hand_card == self.selected_hand_card:
                        self.selected_hand_card = None
                    else:
                        self.selected_hand_card = hand_card
                    self.app.resources.play_sound("card_pickup")
                
                # Check table cards
                table_card = self.get_card_at_position(pos, area="table")
                if table_card:
                    if table_card in self.selected_table_cards:
                        self.selected_table_cards.remove(table_card)
                    else:
                        self.selected_table_cards.add(table_card)
                    self.app.resources.play_sound("card_pickup")
            
            # Handle message log scrolling
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                # Check scroll button clicks
                if hasattr(self, 'scroll_up_rect') and self.scroll_up_rect.collidepoint(pos):
                    self.message_scroll_offset = max(0, self.message_scroll_offset - 1)
                elif hasattr(self, 'scroll_down_rect') and self.scroll_down_rect.collidepoint(pos):
                    self.message_scroll_offset += 1  # Limit is checked in draw_message_log
                
                # Check mouse wheel scrolling over message log
                if hasattr(self, 'scrollbar_rect') and self.message_log_rect.collidepoint(pos):
                    if event.button == 4:  # Scroll up
                        self.message_scroll_offset = max(0, self.message_scroll_offset - 1)
                    elif event.button == 5:  # Scroll down
                        self.message_scroll_offset += 1
    
    def update(self):
        """Update game state"""
        # Update player hands from game state
        self.update_player_hands()
        
        # Update current player
        if self.env:
            self.current_player_id = self.env.current_player
        
        # Update animations
        self.update_animations()
        
        # AGGIUNGERE QUESTO CODICE: Aggiorna local_player_id dal network se necessario
        if self.app.game_config.get("mode") == "online_multiplayer" and not self.app.game_config.get("is_host"):
            if self.app.network and self.app.network.player_id is not None:
                old_player_id = self.local_player_id
                if self.local_player_id != self.app.network.player_id:
                    print(f"Aggiornamento player_id: da {self.local_player_id} a {self.app.network.player_id}")
                    self.local_player_id = self.app.network.player_id
                    # Aggiorna anche il nome del giocatore
                    self.players[self.local_player_id].name = "You"
                    
                    # If player ID changed, recalculate visual positions
                    if old_player_id != self.local_player_id:
                        self.setup_player_perspective()
                        # Force layout recalculation with new perspective
                        self.setup_layout()
        
        # Handle AI turns
        self.handle_ai_turns()
        
        # Handle network updates
        self.handle_network_updates()
        
        # Check for game over
        if self.env and not self.game_over:
            self.check_game_over()
        
        # Gestione dello stato del server in modalità online
        if self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host"):
            online_type = self.app.game_config.get("online_type", "all_human")
            ip_address = get_local_ip()
            
            if online_type == "all_human":
                # In modalità "All Human" serve aspettare 3 giocatori umani
                needed_players = 3
                if self.app.network and len(self.app.network.clients) < needed_players:
                    # Aggiorna il messaggio con il numero di giocatori connessi
                    connected_count = len(self.app.network.clients)
                    self.status_message = f"Server attivo su {ip_address}:5555 | {connected_count}/{needed_players} giocatori connessi"
                    
                    # In attesa di altri giocatori, blocca il gioco finché non si connettono tutti
                    self.waiting_for_other_player = True
                elif self.app.network and len(self.app.network.clients) == needed_players and self.waiting_for_other_player:
                    # Tutti i giocatori sono connessi, sblocca il gioco
                    self.waiting_for_other_player = False
                    self.status_message = "Tutti i giocatori sono connessi. Inizia il gioco!"
            
            elif online_type == "team_vs_ai":
                # In modalità "Team vs AI" serve aspettare solo 1 giocatore umano
                needed_players = 1
                if self.app.network and len(self.app.network.clients) < needed_players:
                    # Aggiorna il messaggio con il numero di giocatori connessi
                    connected_count = len(self.app.network.clients)
                    self.status_message = f"Server attivo su {ip_address}:5555 | {connected_count}/{needed_players} giocatori connessi"
                    
                    # In attesa di altri giocatori, blocca il gioco finché non si connettono tutti
                    self.waiting_for_other_player = True
                elif self.app.network and len(self.app.network.clients) == needed_players and self.waiting_for_other_player:
                    # Configura i giocatori AI per la squadra avversaria (giocatori 1 e 3)
                    self.setup_team_vs_ai_online()
                    # Sblocca il gioco
                    self.waiting_for_other_player = False
                    self.status_message = "Partner connesso. Inizia il gioco!"
        
        # Add a null check before accessing network attributes
        if hasattr(self, 'app') and hasattr(self.app, 'network') and self.app.network:
            while self.app.network.message_queue:
                message = self.app.network.message_queue.popleft()
                self.messages.append(message)
                    
    def setup_team_vs_ai_online(self):
        """Configura i giocatori AI per la modalità Team vs AI online"""
        # Configura i giocatori AI per la squadra avversaria (giocatori 1 e 3)
        difficulty = self.app.game_config.get("difficulty", 1)
        
        # Crea i controller AI per i giocatori 1 e 3
        for player_id in [1, 3]:
            # I giocatori AI appartengono al team 1
            team_id = 1
            self.ai_controllers[player_id] = DQNAgent(team_id=team_id)
            
            # Carica il checkpoint se disponibile
            checkpoint_path = f"scopone_checkpoint_team{team_id}.pth"
            if os.path.exists(checkpoint_path):
                self.ai_controllers[player_id].load_checkpoint(checkpoint_path)
            
            # Imposta il livello di casualità in base alla difficoltà
            if difficulty == 0:  # Easy
                self.ai_controllers[player_id].epsilon = 0.3
            elif difficulty == 1:  # Medium
                self.ai_controllers[player_id].epsilon = 0.1
            else:  # Hard
                self.ai_controllers[player_id].epsilon = 0.01
            
            # Aggiorna lo stato dei giocatori
            self.players[player_id].is_ai = True
            self.players[player_id].name = f"AI {player_id}"
        
        # Aggiorna lo stato del giocatore partner
        self.players[2].is_human = True
        self.players[2].is_ai = False
        self.players[2].name = "Partner"
    
    def update_animations(self):
        """Update and clean up animations"""
        # Update existing animations
        for anim in self.animations[:]:
            if anim.update():
                self.animations.remove(anim)
    
    def handle_ai_turns(self):
        """Handle turns for AI-controlled players"""
        if not self.env or self.game_over or self.animations:
            return
            
        # Check if it's AI's turn
        current_player = self.players[self.current_player_id]
        
        if current_player.is_ai and not self.ai_thinking:
            # Start AI thinking timer
            self.ai_thinking = True
            self.ai_move_timer = pygame.time.get_ticks()
            self.status_message = f"{current_player.name} is thinking..."
            return
            
        # Process AI move after a delay
        if self.ai_thinking:
            current_time = pygame.time.get_ticks()
            # Adjust delay based on difficulty (faster for hard, slower for easy)
            delay = 2000 if self.ai_difficulty == 0 else 1000 if self.ai_difficulty == 1 else 500
            if current_time - self.ai_move_timer > delay:
                self.make_ai_move()
                self.ai_thinking = False
    
    def handle_network_updates(self):
        """Handle network updates for online games with improved state synchronization"""
        if not self.app.network:
            return
            
        # Handle connection loss with reconnection attempt
        if not self.app.network.connected:
            self.status_message = "Connessione persa. Tentativo di riconnettersi..."
            
            # For clients (not host), try to reconnect
            if not self.app.network.is_host:
                if self.app.network.connect_to_server():
                    self.status_message = "Riconnessione avvenuta con successo!"
                else:
                    self.status_message = "Impossibile riconnettersi al server."
                    self.waiting_for_other_player = True
            return
            
        # Process new messages
        while self.app.network.message_queue:
            message = self.app.network.message_queue.popleft()
            self.messages.append(message)
            # Limit messages to 100
            if len(self.messages) > 100:
                self.messages.pop(0)
        
        # If host, check for moves and update game state
        if self.app.network.is_host:
            while self.app.network.move_queue:
                player_id, move = self.app.network.move_queue.popleft()
                
                # Skip if not player's turn
                if player_id != self.env.current_player:
                    continue
                    
                # Decode the move to create animations
                card_played, cards_captured = decode_action(move)
                
                # Create animations for the move
                self.create_move_animations(card_played, cards_captured)
                
                # Make the move
                try:
                    _, _, done, info = self.env.step(move)
                    if done:
                        self.game_over = True
                        if "score_breakdown" in info:
                            self.final_breakdown = info["score_breakdown"]
                except Exception as e:
                    print(f"Error processing move: {e}")
                
                # Broadcast updated game state including current player
                self.app.network.game_state = self.env.game_state
                # Add current player to the game state for synchronization
                self.app.network.game_state['current_player'] = self.env.current_player
                self.app.network.broadcast_game_state()
        
        # If client, update game state from network
        else:
            # Save current state to detect changes
            old_state = None
            old_current_player = None
            if self.env and hasattr(self.env, 'game_state'):
                old_state = self.env.game_state.copy() if self.env.game_state else None
                old_current_player = self.env.current_player if hasattr(self.env, 'current_player') else None
            
            if self.app.network.game_state:
                # Store the new game state
                new_state = self.app.network.game_state
                
                # Extract and remove current player from game state if present
                new_current_player = new_state.pop('current_player', None)
                
                # Update local game state
                if self.env:
                    self.env.game_state = new_state
                    
                    # Update current player if provided
                    if new_current_player is not None:
                        self.env.current_player = new_current_player
                        self.current_player_id = new_current_player
                        print(f"Updated current player to {new_current_player}")
                    
                    # Check if there was a state change and create animations if needed
                    if old_state and old_state != new_state:
                        # Check if cards were played or captured
                        if old_state.get('table') != new_state.get('table') or \
                        old_state.get('captured_squads') != new_state.get('captured_squads') or \
                        old_current_player != new_current_player:
                        
                            # Compare tables to identify played card and captured cards
                            self.detect_and_animate_changes(old_state, new_state)
                            
                            # Unlock waiting state if player was waiting
                            self.waiting_for_other_player = False
                        
                        # Check if game is over
                        if all(len(new_state["hands"][p]) == 0 for p in range(4)):
                            self.game_over = True
                            from rewards import compute_final_score_breakdown
                            self.final_breakdown = compute_final_score_breakdown(new_state)

    
    def detect_and_animate_changes(self, old_state, new_state):
        """Enhanced version to better detect and animate changes"""
        # Skip if no old state
        if not old_state or not new_state:
            return
        
        # Get tables
        old_table = old_state.get('table', [])
        new_table = new_state.get('table', [])
        
        # Find played card - card that moved from a hand to the table
        played_card = None
        player_id = None
        
        # Check each player's hand for a card that disappeared
        for p_id in range(4):
            old_hand = old_state.get('hands', {}).get(p_id, [])
            new_hand = new_state.get('hands', {}).get(p_id, [])
            
            # Convert lists to sets for easier comparison
            old_hand_set = set(tuple(card) for card in old_hand)
            new_hand_set = set(tuple(card) for card in new_hand)
            
            # Find cards that were in old hand but not in new hand
            missing_cards = old_hand_set - new_hand_set
            
            for card in missing_cards:
                card = list(card)  # Convert back to list for further processing
                # Check if this card appeared on the table
                if card in new_table and card not in old_table:
                    played_card = card
                    player_id = p_id
                    break
        
        # Find captured cards - cards that were on the old table but not on the new table
        captured_cards = []
        for card in old_table:
            if card not in new_table:
                captured_cards.append(card)
        
        if played_card:
            print(f"Detected move: Player {player_id} played {played_card} and captured {captured_cards}")
            
            # Only create animations if the player who moved is not the local player
            # (local player animations are created when they make the move)
            if player_id != self.local_player_id:
                # Create animation for the played card and captures
                self.create_move_animations(played_card, captured_cards, player_id)
                
                # Play sound
                self.app.resources.play_sound("card_play")
        else:
            print("Changes detected but couldn't determine the exact move")
            # Log table changes for debugging
            print(f"Old table: {old_table}")
            print(f"New table: {new_table}")
    
    def check_game_over(self):
        """Check if the game is over"""
        if not self.env:
            return
            
        # Check if all hands are empty
        gs = self.env.game_state
        is_game_over = all(len(gs["hands"][p]) == 0 for p in range(4))
        
        if is_game_over:
            self.game_over = True
            
            # Calculate final score
            from rewards import compute_final_score_breakdown
            self.final_breakdown = compute_final_score_breakdown(gs)
            
            # Determine winner
            team0_score = self.final_breakdown[0]["total"]
            team1_score = self.final_breakdown[1]["total"]
            
            if team0_score > team1_score:
                self.status_message = "Team 0 wins!"
                self.app.resources.play_sound("win" if self.local_player_id in [0, 2] else "lose")
            elif team1_score > team0_score:
                self.status_message = "Team 1 wins!"
                self.app.resources.play_sound("win" if self.local_player_id in [1, 3] else "lose")
            else:
                self.status_message = "It's a tie!"
    
    def get_card_at_position(self, pos, area="hand"):
        """Get the card at a position with visual perspective handling"""
        width = self.app.window_width
        card_width = int(width * 0.078)
        card_height = int(card_width * 1.5)
        
        if area == "hand":
            # Determine active player based on game mode and turn
            current_player = None
            mode = self.app.game_config.get("mode")
            
            # For all modes, start with the current player whose turn it is
            if self.is_current_player_controllable():
                current_player = self.players[self.current_player_id]
            else:
                # If not controllable, just check the local player's hand
                current_player = self.players[self.local_player_id]
            
            # Get the player's hand and hand area
            hand = current_player.hand_cards
            hand_rect = current_player.hand_rect
            
            if not hand:
                return None
            
            # Get visual position of the player
            visual_pos = self.get_visual_position(current_player.player_id)
            is_horizontal = visual_pos in [0, 2]  # Bottom or top
            
            if is_horizontal:
                # Horizontal hand (bottom or top player)
                card_spread = card_width * 0.7
                total_width = (len(hand) - 1) * card_spread + card_width
                start_x = hand_rect.centerx - total_width / 2
                
                # Y position depends on visual position
                if visual_pos == 0:  # Bottom
                    base_y = hand_rect.bottom - card_height
                else:  # Top
                    base_y = hand_rect.top
                
                for i, card in enumerate(hand):
                    # Calculate x position for this card
                    x = start_x + i * card_spread
                    
                    # Check if position is within this card
                    card_rect = pygame.Rect(x, base_y, card_width, card_height)
                    if card_rect.collidepoint(pos):
                        return card
            else:
                # Vertical hand (left or right player)
                card_spread = card_height * 0.4
                total_height = (len(hand) - 1) * card_spread + card_height
                start_y = hand_rect.centery - total_height / 2
                
                # X position depends on visual position
                if visual_pos == 1:  # Left
                    base_x = hand_rect.left
                else:  # Right
                    base_x = hand_rect.right - card_width
                
                for i, card in enumerate(hand):
                    # Calculate y position for this card
                    y = start_y + i * card_spread
                    
                    # Check if position is within this card
                    card_rect = pygame.Rect(base_x, y, card_width, card_height)
                    if card_rect.collidepoint(pos):
                        return card
            
            return None
            
        elif area == "table":
            # Table cards are centered and don't change with perspective
            table_cards = self.env.game_state["table"]
            
            if not table_cards:
                return None
            
            # Calculate positions for table layout
            max_spacing = self.table_rect.width * 0.8 / max(len(table_cards), 1)
            card_spacing = min(card_width * 1.1, max_spacing)
            start_x = self.table_rect.centerx - (len(table_cards) * card_spacing) // 2
            y = self.table_rect.centery - card_height // 2
            
            for i, card in enumerate(table_cards):
                x = start_x + i * card_spacing
                
                # Check if position is within this card
                card_rect = pygame.Rect(x, y, card_width, card_height)
                if card_rect.collidepoint(pos):
                    return card
            
            return None
    
    def try_make_move(self):
        """Try to make a move with the selected cards"""
        # Assicurati che sia il turno di un giocatore controllabile
        if not self.is_current_player_controllable():
            self.status_message = "Non è il tuo turno"
            return False
            
        if not self.selected_hand_card:
            self.status_message = "Select a card from your hand first"
            return False
        
        # Encode the action
        action_vec = encode_action(self.selected_hand_card, list(self.selected_table_cards))
        
        # Verify it's a valid action
        valid_actions = self.env.get_valid_actions()
        valid_action = None
        
        for valid_act in valid_actions:
            card, captured = decode_action(valid_act)
            if card == self.selected_hand_card and set(captured) == self.selected_table_cards:
                valid_action = valid_act
                break
        
        if valid_action is None:
            self.status_message = "Invalid move. Try again."
            return False
        
        # Make the move
        try:
            # If online multiplayer, send move to server/other players
            if self.app.game_config.get("mode") == "online_multiplayer":
                self.app.network.send_move(valid_action)
                if not self.app.network.is_host:
                    # Client waits for server to update game state
                    self.waiting_for_other_player = True
                    return True
            
            # Create animations for the move
            self.create_move_animations(self.selected_hand_card, self.selected_table_cards)
            
            # Execute the move
            _, _, done, info = self.env.step(valid_action)
            
            # Play sound
            self.app.resources.play_sound("card_play")
            
            # If game is over, set final state
            if done:
                self.game_over = True
                if "score_breakdown" in info:
                    self.final_breakdown = info["score_breakdown"]
            
            # Reset selection
            self.selected_hand_card = None
            self.selected_table_cards.clear()
            
            return True
        except Exception as e:
            print(f"Error making move: {e}")
            self.status_message = "Error making move"
            return False
    
    def make_ai_move(self):
        """Make a move for the current AI player"""
        current_player = self.players[self.current_player_id]
        
        if not current_player.is_ai or self.current_player_id not in self.ai_controllers:
            return
        
        # Get AI controller
        ai = self.ai_controllers[self.current_player_id]
        
        # Get observation and valid actions
        obs = self.env._get_observation(self.current_player_id)
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            return
        
        # Choose action using the AI agent
        action = ai.pick_action(obs, valid_actions, self.env)
        
        # Get the card played and cards captured for animation
        card_played, cards_captured = decode_action(action)
        
        # Create animations
        self.create_move_animations(card_played, cards_captured)
        
        # Execute the action
        _, _, done, info = self.env.step(action)
        
        # Play sound
        self.app.resources.play_sound("card_play")
        
        # Update game state
        if done:
            self.game_over = True
            if "score_breakdown" in info:
                self.final_breakdown = info["score_breakdown"]
    
    def create_move_animations(self, card_played, cards_captured, source_player_id=None):
        """Create animations for a move"""
        # Get player ID - use current player if no source player specified
        player_id = source_player_id if source_player_id is not None else self.current_player_id
        current_player = self.players[player_id]
        
        # Start position depends on player position
        if player_id == self.local_player_id:
            # Calculate position in hand
            hand = current_player.hand_cards
            try:
                card_index = hand.index(card_played)
                hand_rect = current_player.hand_rect
                
                # Calculate position with current dimensions
                width = self.app.window_width
                card_width = int(width * 0.078)
                center_x = hand_rect.centerx
                card_spread = card_width * 0.7
                
                # Calculate position
                start_x = center_x + (card_index - len(hand) // 2) * card_spread
                start_y = hand_rect.bottom - CARD_HEIGHT
                
                start_pos = (start_x + CARD_WIDTH // 2, start_y + CARD_HEIGHT // 2)
            except ValueError:
                # Card not found in hand (should not happen)
                start_pos = current_player.hand_rect.center
        else:
            # Card comes from another player's hand
            start_pos = current_player.hand_rect.center
        
        # Calculate table center for destination
        table_center = self.table_rect.center
        
        # Create animation for played card
        played_anim = CardAnimation(
            card=card_played,
            start_pos=start_pos,
            end_pos=table_center,
            duration=20,
            scale_start=1.0,
            scale_end=1.0,
            rotation_start=0,
            rotation_end=0  # No rotation
        )
        self.animations.append(played_anim)
        
        # Create animations for captured cards
        table_cards = self.env.game_state["table"]
        
        # Calculate card positions in table layout with current dimensions
        if table_cards or cards_captured:
            width = self.app.window_width
            card_width = int(width * 0.078)
            
            # We need to calculate positions for captured cards that might no longer be on the table
            # Use the current table plus captured cards to determine original positions
            original_table = table_cards.copy()
            for card in cards_captured:
                if card not in original_table:
                    original_table.append(card)
            
            max_spacing = self.table_rect.width * 0.8 / max(len(original_table), 1)
            card_spacing = min(card_width * 1.1, max_spacing)
            start_x = self.table_rect.centerx - (len(original_table) * card_spacing) // 2
            table_y = self.table_rect.centery - CARD_HEIGHT // 2
            
            # Calculate pile position for the capturing team
            team_id = current_player.team_id
            pile_pos = (self.app.window_width * 0.05, self.app.window_height // 2 + (team_id * 0.2 - 0.1) * self.app.window_height)
            
            for card in cards_captured:
                try:
                    # Find card position in the original table
                    card_index = original_table.index(card)
                    card_x = start_x + card_index * card_spacing
                    card_pos = (card_x + CARD_WIDTH // 2, table_y + CARD_HEIGHT // 2)
                    
                    capture_anim = CardAnimation(
                        card=card,
                        start_pos=card_pos,
                        end_pos=pile_pos,
                        duration=25,
                        delay=25,  # Longer delay to ensure played card is shown first
                        scale_start=1.0,
                        scale_end=0.8,
                        rotation_start=0,
                        rotation_end=0  # No rotation
                    )
                    self.animations.append(capture_anim)
                except ValueError:
                    # Card not found on table (should not happen)
                    print(f"Warning: Could not find card {card} in original table")
    
    def draw(self, surface):
        """Draw the game screen"""
        # Draw background
        surface.blit(self.app.resources.background, (0, 0))
        
        # Draw table
        pygame.draw.ellipse(surface, TABLE_GREEN, 
                           self.table_rect.inflate(50, 30))
        pygame.draw.ellipse(surface, DARK_GREEN, 
                           self.table_rect.inflate(50, 30), 5)
        
        # Draw players
        self.draw_players(surface)
        
        # Draw table cards
        self.draw_table_cards(surface)
        
        # Draw capture piles
        self.draw_capture_piles(surface)
        
        # Draw animations
        self.draw_animations(surface)
        
        # Draw status info
        self.draw_status_info(surface)
        
        # Draw buttons
        self.new_game_button.draw(surface)
        
        # Draw confirm button if current player is controllable
        if self.env and self.is_current_player_controllable() and not self.game_over:
            self.confirm_button.draw(surface)
        
        # Draw message log
        self.draw_message_log(surface)
        
        # Draw game over screen if game is over
        if self.game_over and self.final_breakdown:
            self.draw_game_over(surface)
    
    def draw_players(self, surface):
        """Draw all players' hands and info with proper perspective handling"""
        for player in self.players:
            self.draw_player_info(surface, player)
            
            # Determine if player's hand should be visible
            mode = self.app.game_config.get("mode")
            
            # Always show local player's hand
            if player.player_id == self.local_player_id:
                self.draw_player_hand(surface, player)
            
            # For team_vs_ai, show partner's hand when it's their turn
            elif mode == "team_vs_ai" and player.player_id == 2 and self.current_player_id == 2:
                self.draw_player_hand(surface, player)
                
            # For local multiplayer, show current player's hand
            elif mode == "local_multiplayer" and player.player_id == self.current_player_id:
                self.draw_player_hand(surface, player)
                
            # Otherwise show hidden hand
            else:
                self.draw_player_hidden_hand(surface, player)
    
    def draw_player_info(self, surface, player):
        """Draw player information within the existing avatar box"""
        # Get screen dimensions
        width = self.app.window_width
        height = self.app.window_height
        
        # Draw avatar background (team-colored box)
        avatar_color = LIGHT_BLUE if player.team_id == 0 else HIGHLIGHT_RED
        pygame.draw.rect(surface, avatar_color, player.avatar_rect, border_radius=10)
        
        # Draw border if it's current player's turn
        if player.player_id == self.current_player_id:
            pygame.draw.rect(surface, GOLD, player.avatar_rect.inflate(6, 6), 3, border_radius=12)
        
        # Calculate text positions inside the avatar box
        # Create a smaller font for fitting text in the box
        info_font = pygame.font.SysFont(None, int(height * 0.022))
        
        # Player name at top
        name_text = player.name
        if player.is_ai:
            name_text += " (AI)"
        
        name_surf = info_font.render(name_text, True, WHITE)
        name_rect = name_surf.get_rect(
            midtop=(player.avatar_rect.centerx, player.avatar_rect.top + 5)
        )
        surface.blit(name_surf, name_rect)
        
        # Team info in middle
        team_text = f"Team {player.team_id}"
        team_surf = info_font.render(team_text, True, WHITE)
        team_rect = team_surf.get_rect(
            center=(player.avatar_rect.centerx, player.avatar_rect.centery)
        )
        surface.blit(team_surf, team_rect)
        
        # Card count at bottom
        count_text = f"Cards: {len(player.hand_cards)}"
        count_surf = info_font.render(count_text, True, WHITE)
        count_rect = count_surf.get_rect(
            midbottom=(player.avatar_rect.centerx, player.avatar_rect.bottom - 5)
        )
        surface.blit(count_surf, count_rect)
    
    def draw_player_hand(self, surface, player):
        """Draw the player's hand with card faces visible and properly centered"""
        hand = player.hand_cards
        if not hand:
            return
        
        # Calculate the current card width based on window size
        width = self.app.window_width
        card_width = int(width * 0.078)
        card_height = int(card_width * 1.5)
        
        # Get position based on visual perspective
        visual_pos = self.get_visual_position(player.player_id)
        hand_rect = player.hand_rect
        
        # Display variables depend on whether this is horizontal or vertical orientation
        is_horizontal = visual_pos in [0, 2]  # Bottom or top
        
        if is_horizontal:
            # Horizontal hand layout (bottom or top)
            center_x = hand_rect.centerx
            card_spread = card_width * 0.7
            total_width = (len(hand) - 1) * card_spread + card_width
            start_x = center_x - (total_width / 2)
            
            # Y position depends on whether it's top or bottom
            if visual_pos == 0:  # Bottom
                base_y = hand_rect.bottom - card_height
            else:  # Top
                base_y = hand_rect.top
            
            # Debug output
            print(f"Drawing hand for player {player.player_id} at visual pos {visual_pos}. Cards: {len(hand)}")
            print(f"Hand rect: {hand_rect}, Start X: {start_x}, Base Y: {base_y}")
            
            for i, card in enumerate(hand):
                # Calculate x position for this card
                x = start_x + i * card_spread
                
                # Get card image
                card_img = self.app.resources.get_card_image(card)
                
                # Create card rect
                card_rect = pygame.Rect(x, base_y, card_width, card_height)
                
                # Highlight selected card
                if card == self.selected_hand_card:
                    # Draw selection border
                    highlight_rect = card_rect.copy()
                    pygame.draw.rect(surface, HIGHLIGHT_BLUE, highlight_rect.inflate(10, 10), 3, border_radius=5)
                    
                    # Raise selected card for bottom player, lower for top player
                    if visual_pos == 0:
                        card_rect.move_ip(0, -15)
                    else:
                        card_rect.move_ip(0, 15)
                
                # Draw card
                surface.blit(card_img, card_rect)
        else:
            # Vertical hand layout (left or right)
            center_y = hand_rect.centery
            card_spread = card_height * 0.4  # Less overlap for vertical cards
            total_height = (len(hand) - 1) * card_spread + card_height
            start_y = center_y - (total_height / 2)
            
            # X position depends on whether it's left or right
            if visual_pos == 1:  # Left
                base_x = hand_rect.left
            else:  # Right
                base_x = hand_rect.right - card_width
            
            for i, card in enumerate(hand):
                # Calculate y position for this card
                y = start_y + i * card_spread
                
                # Get card image
                card_img = self.app.resources.get_card_image(card)
                
                # Create card rect
                card_rect = pygame.Rect(base_x, y, card_width, card_height)
                
                # Highlight selected card
                if card == self.selected_hand_card:
                    # Draw selection border
                    highlight_rect = card_rect.copy()
                    pygame.draw.rect(surface, HIGHLIGHT_BLUE, highlight_rect.inflate(10, 10), 3, border_radius=5)
                    
                    # Move selected card outward
                    if visual_pos == 1:
                        card_rect.move_ip(15, 0)
                    else:
                        card_rect.move_ip(-15, 0)
                
                # Draw card
                surface.blit(card_img, card_rect)
    
    def draw_player_hidden_hand(self, surface, player):
        """Draw another player's hand with card backs"""
        hand = player.hand_cards
        if not hand:
            return
        
        # Get card back image for this player's team
        card_back = self.app.resources.get_card_back(player.team_id)
        
        # Left or right player (vertical layout)
        if player.player_id in [1, 3]:
            # Calculate positions
            card_spacing = min(20, player.hand_rect.height / len(hand))
            start_y = player.hand_rect.centery - (len(hand) * card_spacing) // 2
            
            for i in range(len(hand)):
                y = start_y + i * card_spacing
                
                # Rotate card back for vertical display
                rotated_back = pygame.transform.rotate(card_back, 90)
                
                # Draw card back
                surface.blit(rotated_back, (player.hand_rect.x, y))
        
        # Top or bottom player (horizontal layout)
        else:
            # Calculate positions
            card_spacing = min(20, player.hand_rect.width / len(hand))
            start_x = player.hand_rect.centerx - (len(hand) * card_spacing) // 2
            
            for i in range(len(hand)):
                x = start_x + i * card_spacing
                
                # Draw card back
                surface.blit(card_back, (x, player.hand_rect.y))
    
    def draw_table_cards(self, surface):
        """Draw the cards on the table without any rotation"""
        if not self.env:
            return
                
        table_cards = self.env.game_state["table"]
        if not table_cards:
            # Draw "No cards on table" text
            text_surf = self.info_font.render("No cards on table", True, WHITE)
            text_rect = text_surf.get_rect(center=self.table_rect.center)
            surface.blit(text_surf, text_rect)
            return
        
        # Get the current card size based on window size
        width = self.app.window_width
        card_width = int(width * 0.078)
        card_height = int(card_width * 1.5)
        
        # Calculate positions for table layout - use table width to determine spacing
        max_spacing = self.table_rect.width * 0.8 / max(len(table_cards), 1)
        card_spacing = min(card_width * 1.1, max_spacing)
        start_x = self.table_rect.centerx - (len(table_cards) * card_spacing) // 2
        y = self.table_rect.centery - card_height // 2
        
        for i, card in enumerate(table_cards):
            x = start_x + i * card_spacing
            
            # Get card image
            card_img = self.app.resources.get_card_image(card)
            
            # Create rect for the card at its position (no rotation)
            card_rect = pygame.Rect(x, y, card_width, card_height)
            
            # Highlight selected table cards
            if card in self.selected_table_cards:
                pygame.draw.rect(surface, HIGHLIGHT_RED, card_rect.inflate(10, 10), 3, border_radius=5)
            
            # Draw card (without rotation)
            surface.blit(card_img, card_rect)
    
    def draw_capture_piles(self, surface):
        """Draw the capture piles for each team with updated positioning"""
        if not self.env:
            return
                
        captured = self.env.game_state["captured_squads"]
        width = self.app.window_width
        height = self.app.window_height
        
        # Calculate pile positions based on the updated layout
        # Top center positioning
        pile0_x = width // 2 - int(width * 0.22)  # Left of center
        pile0_y = height * 0.02                   # Top
        pile_width = int(width * 0.12)
        pile_height = int(height * 0.13)
        
        pile1_x = width // 2 + int(width * 0.10)  # Right of center
        pile1_y = height * 0.02                   # Top
        
        # Draw team 0 pile with team-specific color
        team0_count = len(captured[0])
        
        # Draw pile background - BLUE for team 0
        pygame.draw.rect(surface, LIGHT_BLUE, 
                        (pile0_x, pile0_y, pile_width, pile_height), 
                        border_radius=5)
        
        # Draw text
        text0 = f"Team 0: {team0_count} cards"
        text0_surf = self.small_font.render(text0, True, WHITE)
        surface.blit(text0_surf, (pile0_x + 10, pile0_y + 5))
        
        # Draw a few cards stacked
        max_display = min(5, team0_count)
        for i in range(max_display):
            # Calculate slightly offset position
            x = pile0_x + 10 + i * 5
            y = pile0_y + 30 + i * 5
            
            # Draw a mini card
            mini_width = int(width * 0.04)
            mini_height = int(height * 0.08)
            mini_card = pygame.Surface((mini_width, mini_height))
            mini_card.fill(WHITE)
            pygame.draw.rect(mini_card, DARK_BLUE, (0, 0, mini_width, mini_height), 2)
            surface.blit(mini_card, (x, y))
        
        # Draw team 1 pile with team-specific color
        team1_count = len(captured[1])
        
        # Draw pile background - RED for team 1
        pygame.draw.rect(surface, HIGHLIGHT_RED, 
                        (pile1_x, pile1_y, pile_width, pile_height), 
                        border_radius=5)
        
        # Draw text
        text1 = f"Team 1: {team1_count} cards"
        text1_surf = self.small_font.render(text1, True, WHITE)
        surface.blit(text1_surf, (pile1_x + 10, pile1_y + 5))
        
        # Draw a few cards stacked
        max_display = min(5, team1_count)
        for i in range(max_display):
            # Calculate slightly offset position
            x = pile1_x + 10 + i * 5
            y = pile1_y + 30 + i * 5
            
            # Draw a mini card
            mini_width = int(width * 0.04)
            mini_height = int(height * 0.08)
            mini_card = pygame.Surface((mini_width, mini_height))
            mini_card.fill(WHITE)
            pygame.draw.rect(mini_card, DARK_RED if 'DARK_RED' in globals() else (139, 0, 0), (0, 0, mini_width, mini_height), 2)
            surface.blit(mini_card, (x, y))
    
    def draw_animations(self, surface):
        """Draw all active card animations"""
        for anim in self.animations:
            if anim.current_frame < 0:
                continue  # Animation in delay phase
            
            # Get card image
            card_img = self.app.resources.get_card_image(anim.card)
            
            # Apply scaling
            scale = anim.get_current_scale()
            scaled_img = pygame.transform.scale(
                card_img,
                (int(CARD_WIDTH * scale), int(CARD_HEIGHT * scale))
            )
            
            # Apply rotation
            rotation = anim.get_current_rotation()
            rotated_img = pygame.transform.rotate(scaled_img, rotation)
            
            # Get current position
            pos = anim.get_current_pos()
            
            # Draw card centered at position
            rect = rotated_img.get_rect(center=pos)
            surface.blit(rotated_img, rect)
    
    def draw_status_info(self, surface):
        """Draw game status information"""
        width = self.app.window_width
        height = self.app.window_height
        
        # Draw status message
        if self.status_message:
            msg_surf = self.info_font.render(self.status_message, True, WHITE)
            msg_rect = msg_surf.get_rect(center=(width // 2, height * 0.026))
            surface.blit(msg_surf, msg_rect)
        
        # Draw current player indicator
        if self.env:
            current_player = self.players[self.current_player_id]
            turn_text = f"Current turn: {current_player.name} (Team {current_player.team_id})"
            turn_surf = self.info_font.render(turn_text, True, WHITE)
            turn_rect = turn_surf.get_rect(center=(width // 2, height * 0.065))
            surface.blit(turn_surf, turn_rect)
            
            # Draw difficulty info if playing against AI
            if self.app.game_config.get("mode") in ["single_player", "team_vs_ai"]:
                diff_text = "AI Difficulty: "
                if self.ai_difficulty == 0:
                    diff_text += "Easy"
                    diff_color = DARK_GREEN
                elif self.ai_difficulty == 1:
                    diff_text += "Medium"
                    diff_color = LIGHT_BLUE
                else:
                    diff_text += "Hard"
                    diff_color = HIGHLIGHT_RED
                
                diff_surf = self.small_font.render(diff_text, True, diff_color)
                diff_rect = diff_surf.get_rect(topright=(width - width * 0.02, height * 0.026))
                surface.blit(diff_surf, diff_rect)
    
    def draw_message_log(self, surface):
        """Draw message log with scrolling support"""
        # Draw background
        pygame.draw.rect(surface, DARK_BLUE, self.message_log_rect, border_radius=5)
        pygame.draw.rect(surface, LIGHT_BLUE, self.message_log_rect, 2, border_radius=5)
        
        # Draw title
        title_surf = self.small_font.render("Messages", True, WHITE)
        title_rect = title_surf.get_rect(midtop=(self.message_log_rect.centerx, self.message_log_rect.top + 5))
        surface.blit(title_surf, title_rect)
        
        # Calcola area utile per i messaggi
        content_rect = pygame.Rect(
            self.message_log_rect.left + 5,
            self.message_log_rect.top + 25,
            self.message_log_rect.width - 30,  # Spazio per la scrollbar
            self.message_log_rect.height - 30
        )
        
        # Recupera tutti i messaggi (non limitandoli più a 5)
        all_messages = []
        
        # Se siamo in modalità online e host, mostriamo istruzioni di connessione
        if self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host"):
            # Mostra istruzioni per connettere altri giocatori in modo più chiaro
            local_ip = get_local_ip()
            public_ip = self.app.network.public_ip if hasattr(self.app.network, 'public_ip') else "N/A"
            
            all_messages.append(("Per giocare in LAN:", GOLD))
            all_messages.append((f"IP Locale: {local_ip}", WHITE))
            all_messages.append(("Per giocare su Internet:", GOLD))
            all_messages.append((f"IP Pubblico: {public_ip}", WHITE))
            all_messages.append(("Porta: 5555", WHITE))
            
            # Mostra quanti giocatori sono necessari in base alla modalità
            online_type = self.app.game_config.get("online_type", "all_human")
            if online_type == "all_human":
                all_messages.append(("Attendo 3 giocatori", GOLD))
            else:  # "team_vs_ai"
                all_messages.append(("Attendo 1 partner", GOLD))
        
        # Aggiungi tutti i messaggi dal log
        for msg in self.messages:
            all_messages.append((msg, WHITE))
        
        # Calcola il massimo numero di messaggi visibili nell'area
        msg_height = int(self.app.window_height * 0.023) + 5  # Altezza font + spaziatura
        max_visible = min(len(all_messages), int(content_rect.height / msg_height))
        
        # Limita lo scrolling
        total_messages = len(all_messages)
        max_offset = max(0, total_messages - max_visible)
        self.message_scroll_offset = min(max_offset, max(0, self.message_scroll_offset))
        
        # Disegna i messaggi visibili
        for i in range(max_visible):
            idx = i + self.message_scroll_offset
            if idx < len(all_messages):
                msg_text, msg_color = all_messages[idx]
                msg_surf = self.small_font.render(msg_text, True, msg_color)
                
                # Tronca se troppo lungo
                if msg_surf.get_width() > content_rect.width - 10:
                    truncated = msg_text[:30] + "..."
                    msg_surf = self.small_font.render(truncated, True, msg_color)
                
                y_pos = content_rect.top + i * msg_height
                surface.blit(msg_surf, (content_rect.left, y_pos))
        
        # Disegna la scrollbar se necessaria
        if total_messages > max_visible:
            scrollbar_rect = pygame.Rect(
                self.message_log_rect.right - 20,
                content_rect.top,
                15,
                content_rect.height
            )
            pygame.draw.rect(surface, LIGHT_GRAY, scrollbar_rect, border_radius=7)
            
            # Calcola la posizione e dimensione del cursore di scorrimento
            thumb_height = max(30, scrollbar_rect.height * max_visible / total_messages)
            thumb_pos = scrollbar_rect.top + (scrollbar_rect.height - thumb_height) * self.message_scroll_offset / max_offset
            
            thumb_rect = pygame.Rect(
                scrollbar_rect.left,
                thumb_pos,
                scrollbar_rect.width,
                thumb_height
            )
            pygame.draw.rect(surface, WHITE, thumb_rect, border_radius=7)
            
            # Disegna pulsanti di scorrimento
            up_arrow = pygame.Rect(scrollbar_rect.left, scrollbar_rect.top - 20, 15, 15)
            down_arrow = pygame.Rect(scrollbar_rect.left, scrollbar_rect.bottom + 5, 15, 15)
            
            pygame.draw.rect(surface, LIGHT_GRAY, up_arrow, border_radius=3)
            pygame.draw.rect(surface, LIGHT_GRAY, down_arrow, border_radius=3)
            
            # Disegna triangoli per le frecce
            pygame.draw.polygon(surface, WHITE, [
                (up_arrow.centerx, up_arrow.top + 3),
                (up_arrow.left + 3, up_arrow.bottom - 3),
                (up_arrow.right - 3, up_arrow.bottom - 3)
            ])
            
            pygame.draw.polygon(surface, WHITE, [
                (down_arrow.centerx, down_arrow.bottom - 3),
                (down_arrow.left + 3, down_arrow.top + 3),
                (down_arrow.right - 3, down_arrow.top + 3)
            ])
            
            # Salva i rettangoli per il rilevamento dei clic
            self.scroll_up_rect = up_arrow
            self.scroll_down_rect = down_arrow
            self.scrollbar_rect = scrollbar_rect
    
    def draw_game_over(self, surface):
        """Draw responsive game over screen with results"""
        # Get current dimensions
        width = self.app.window_width
        height = self.app.window_height
        
        # Create semi-transparent overlay
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))
        
        # Draw results panel - sized based on window dimensions
        panel_width = width * 0.7
        panel_height = height * 0.7
        panel_rect = pygame.Rect(
            width // 2 - panel_width // 2, 
            height // 2 - panel_height // 2, 
            panel_width, 
            panel_height
        )
        
        pygame.draw.rect(surface, DARK_BLUE, panel_rect, border_radius=10)
        pygame.draw.rect(surface, GOLD, panel_rect, 4, border_radius=10)
        
        # Draw title with responsive font sizes
        title_font = pygame.font.SysFont(None, int(height * 0.06))
        detail_font = pygame.font.SysFont(None, int(height * 0.03))
        category_font = pygame.font.SysFont(None, int(height * 0.04))
        
        title_surf = title_font.render("Game Over", True, GOLD)
        title_rect = title_surf.get_rect(midtop=(panel_rect.centerx, panel_rect.top + height * 0.03))
        surface.blit(title_surf, title_rect)
        
        # Draw team scores
        team0_score = self.final_breakdown[0]["total"]
        team1_score = self.final_breakdown[1]["total"]
        
        if team0_score > team1_score:
            winner_text = "Team 0 wins!"
            winner_color = LIGHT_BLUE
        elif team1_score > team0_score:
            winner_text = "Team 1 wins!"
            winner_color = HIGHLIGHT_RED
        else:
            winner_text = "It's a tie!"
            winner_color = WHITE
        
        winner_surf = title_font.render(winner_text, True, winner_color)
        winner_rect = winner_surf.get_rect(midtop=(panel_rect.centerx, panel_rect.top + height * 0.1))
        surface.blit(winner_surf, winner_rect)
        
        # Draw score breakdown with responsive positioning
        # Team 0 column (left side)
        team0_title = category_font.render("Team 0 Score", True, LIGHT_BLUE)
        team0_title_rect = team0_title.get_rect(
            topleft=(panel_rect.left + panel_width * 0.05, panel_rect.top + panel_height * 0.25)
        )
        surface.blit(team0_title, team0_title_rect)
        
        score_x = panel_rect.left + width * 0.07
        score_y = team0_title_rect.bottom + height * 0.02
        
        for category, score in self.final_breakdown[0].items():
            if category == "total":
                continue
            text = f"{category.capitalize()}: {score}"
            text_surf = detail_font.render(text, True, WHITE)
            surface.blit(text_surf, (score_x, score_y))
            score_y += height * 0.04
        
        # Total for team 0
        total0_text = f"Total: {self.final_breakdown[0]['total']}"
        total0_surf = category_font.render(total0_text, True, LIGHT_BLUE)
        surface.blit(total0_surf, (score_x, score_y + height * 0.02))
        
        # Team 1 column (right side)
        team1_title = category_font.render("Team 1 Score", True, HIGHLIGHT_RED)
        team1_title_rect = team1_title.get_rect(
            topleft=(panel_rect.centerx + panel_width * 0.05, panel_rect.top + panel_height * 0.25)
        )
        surface.blit(team1_title, team1_title_rect)
        
        score_x = panel_rect.centerx + width * 0.07
        score_y = team1_title_rect.bottom + height * 0.02
        
        for category, score in self.final_breakdown[1].items():
            if category == "total":
                continue
            text = f"{category.capitalize()}: {score}"
            text_surf = detail_font.render(text, True, WHITE)
            surface.blit(text_surf, (score_x, score_y))
            score_y += height * 0.04
        
        # Total for team 1
        total1_text = f"Total: {self.final_breakdown[1]['total']}"
        total1_surf = category_font.render(total1_text, True, HIGHLIGHT_RED)
        surface.blit(total1_surf, (score_x, score_y + height * 0.02))
        
        # Draw new game button
        button_width = width * 0.2
        button_height = height * 0.07
        new_game_button = Button(
            panel_rect.centerx - button_width // 2,
            panel_rect.bottom - button_height - height * 0.03,
            button_width, button_height,
            "New Game",
            DARK_GREEN, WHITE,
            font_size=int(height * 0.03)
        )
        new_game_button.draw(surface)
        
        # Store button rect for click detection
        self.game_over_button_rect = new_game_button.rect
        
    def is_current_player_controllable(self):
        """Verifica se il giocatore corrente può essere controllato dall'utente locale"""
        mode = self.app.game_config.get("mode")
        
        # Per le modalità online
        if mode == "online_multiplayer":
            # Se è il turno del giocatore locale
            if self.current_player_id == self.local_player_id:
                return True
            # Se siamo in modalità team vs AI e il giocatore locale è nella stessa squadra del partner umano
            if self.app.game_config.get("online_type") == "team_vs_ai":
                # In Team vs AI: il giocatore 0 è partner del giocatore 2
                if (self.local_player_id == 0 and self.current_player_id == 2) or \
                (self.local_player_id == 2 and self.current_player_id == 0):
                    return True
            return False
        
        # Per le altre modalità
        elif mode == "team_vs_ai":
            # In modalità team vs AI, entrambi i giocatori 0 e 2 sono controllabili
            return self.current_player_id in [0, 2]
        elif mode == "local_multiplayer":
            # In modalità 4 giocatori umani, tutti i giocatori sono controllabili quando è il loro turno
            return True
        else:
            # Nelle altre modalità, solo il giocatore locale è controllabile
            return self.current_player_id == self.local_player_id

class ScoponeApp:
    """Main application class"""
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Scopone a Coppie")
        
        # Ottieni le dimensioni dello schermo
        display_info = pygame.display.Info()
        available_width = display_info.current_w
        available_height = display_info.current_h
        
        # Imposta dimensioni massime che non superino lo schermo
        # Usa una piccola riduzione (90%) per lasciare spazio per la barra delle applicazioni
        max_width = min(1024, int(available_width * 0.9))
        max_height = min(768, int(available_height * 0.9))
        
        # Aggiorna le costanti globali
        global SCREEN_WIDTH, SCREEN_HEIGHT
        SCREEN_WIDTH = max_width
        SCREEN_HEIGHT = max_height
        
        # Impostazioni della finestra
        flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
        self.window_width = SCREEN_WIDTH
        self.window_height = SCREEN_HEIGHT
        self.clock = pygame.time.Clock()
        
        # Load resources
        self.resources = ResourceManager()
        self.resources.load_resources()
        
        # Set up screens
        self.screens = {
            "mode": GameModeScreen(self),
            "game": GameScreen(self)
        }
        self.current_screen = "mode"
        
        # Game configuration
        self.game_config = {}
        
        # Network manager (for online play)
        self.network = None
        
        # Store initial window size
        self.initial_size = (SCREEN_WIDTH, SCREEN_HEIGHT)

    # Replace the run method in the ScoponeApp class
    def run(self):
        """Main game loop"""
        running = True
        already_entered = False  # Track if we've entered the current screen
        
        while running:
            # Get current screen
            screen = self.screens[self.current_screen]
            
            # Call enter method ONLY when first accessing the screen
            if not already_entered:
                screen.enter()
                already_entered = True
            
            # Handle events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
                    # Get the current window size directly - more reliable
                    current_size = pygame.display.get_surface().get_size()
                    self.handle_resize(current_size)
                
                # Pass remaining events to current screen
                screen.handle_events([event])
            
            # Update and draw current screen
            screen.update()
            screen.draw(self.screen)
            
            # Check if screen is done
            if screen.done:
                # Call exit method
                screen.exit()
                
                # Switch to next screen
                self.current_screen = screen.next_screen
                
                # Reset the entered flag when switching screens
                already_entered = False
                
                # Check if game is over
                if screen.next_screen is None:
                    running = False
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Clean up
        if self.network:
            self.network.close()
        
        pygame.quit()
        sys.exit()


    def handle_resize(self, size):
        """Handle window resize event"""
        # Update window dimensions
        self.window_width, self.window_height = size
        
        # Debug print
        print(f"Resized window to: {self.window_width}x{self.window_height}")
        
        # Scale background image from the original
        if hasattr(self.resources, 'original_background') and self.resources.original_background:
            print("Scaling background from original image")
            # Force the background to scale from the original image
            # Use smoothscale for better quality
            self.resources.background = pygame.transform.smoothscale(
                self.resources.original_background, 
                (self.window_width, self.window_height)
            )
        else:
            print("WARNING: Original background not found, creating new background")
            # Create a new background if the original isn't available
            self.resources.background = pygame.Surface((self.window_width, self.window_height))
            self.resources.background.fill(DARK_BLUE)
        
        # Recalculate layout for all screens
        for screen in self.screens.values():
            if hasattr(screen, 'setup_layout'):
                screen.setup_layout()

if __name__ == "__main__":
    app = ScoponeApp()
    app.run()