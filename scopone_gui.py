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
DARK_RED = (139, 0, 0)
GOLD = (255, 215, 0)
DARK_BLUE = (20, 51, 104)
LIGHT_BLUE = (100, 149, 237)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
# Additional UI colors for status dots
GREEN = (0, 170, 0)
LIGHT_GREEN = (144, 238, 144)
ORANGE = (255, 165, 0)
YELLOW = (255, 215, 0)  # same as GOLD, explicit alias for clarity
RED = (255, 0, 0)
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
        # Usa dimensioni correnti della superficie per essere responsive al resize
        current_width, current_height = surface.get_size()

        # Crea un overlay semi-trasparente che copra l'intera superficie corrente
        overlay = pygame.Surface((current_width, current_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Nero semi-trasparente
        surface.blit(overlay, (0, 0))
        
        # Disegna il cerchio rotante centrato e proporzionale alle dimensioni correnti
        center_x = current_width // 2
        center_y = current_height // 2
        radius = min(current_width, current_height) * 0.1
        
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
        
        # Disegna il messaggio di caricamento con font proporzionale all'altezza corrente
        font = pygame.font.SysFont(None, int(current_height * 0.04))
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
    """Class for card movement animations with phase tracking"""
    def __init__(self, card, start_pos, end_pos, duration, delay=0, 
                 scale_start=1.0, scale_end=1.0, rotation_start=0, rotation_end=0,
                 animation_type="play"):
        self.card = card
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.delay = delay
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.rotation_start = rotation_start
        self.rotation_end = rotation_end
        self.animation_type = animation_type  # "play" per carta giocata, "capture" per cattura
        
        self.current_frame = -delay
        self.done = False
        
        # Aggiungiamo un log quando creiamo una nuova animazione con delay
        #if delay > 0:
            #print(f"DEBUG: Creata animazione {animation_type} per carta {card} con delay {delay}")
    
    def update(self):
        """Aggiorna lo stato dell'animazione e restituisce True se completata"""
        # Se siamo in fase di delay, decrementa il contatore
        if self.current_frame < 0:
            # Debug per tracciare il delay
            #if self.current_frame == -1:
                #print(f"DEBUG: Animazione {self.animation_type} per carta {self.card} inizia (ultimo frame di delay)")
            #elif self.current_frame % 10 == 0:  # Log ogni 10 frame per non intasare la console
                #print(f"DEBUG: Animazione {self.animation_type} per carta {self.card} in delay: {-self.current_frame} frame rimanenti")
                
            self.current_frame += 1
            return False
            
        # Se abbiamo superato la durata, segna come completata
        if self.current_frame >= self.duration:
            # Se non era già segnata come completata, stampa un log
            if not self.done:
                #print(f"DEBUG: Animazione {self.animation_type} per carta {self.card} completata")
                self.done = True
            return True
            
        # Altrimenti, incrementa il frame e continua l'animazione
        self.current_frame += 1
        
        # Debug per tracciare l'animazione attiva
        #if self.current_frame == 0:
            #print(f"DEBUG: Animazione {self.animation_type} per carta {self.card} iniziata attivamente")
        #elif self.current_frame == self.duration - 1:
            #print(f"DEBUG: Animazione {self.animation_type} per carta {self.card} al penultimo frame")
            
        return False
    
    def get_current_pos(self):
        """Calcola la posizione corrente in base al progresso dell'animazione"""
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
        """Calcola la scala corrente in base al progresso dell'animazione"""
        if self.current_frame < 0:
            return self.scale_start
            
        if self.current_frame >= self.duration:
            return self.scale_end
            
        progress = self.current_frame / self.duration
        return self.scale_start + (self.scale_end - self.scale_start) * progress
    
    def get_current_rotation(self):
        """Calcola la rotazione corrente in base al progresso dell'animazione"""
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
        
        # Try to load retro.jpg first
        try:
            retro = pygame.image.load(os.path.join(folder, "retro.jpg"))
            # Use the same retro for both teams
            self.original_card_backs[0] = retro
            self.original_card_backs[1] = retro
            self.card_backs[0] = pygame.transform.scale(retro, (CARD_WIDTH, CARD_HEIGHT))
            self.card_backs[1] = pygame.transform.scale(retro, (CARD_WIDTH, CARD_HEIGHT))
            #print("Caricato retro.jpg per dorso carte")
        except Exception as e:
            print(f"Errore caricamento retro.jpg: {e}")
            # If retro.jpg not found, try team-specific backs
            try:
                back_blue = pygame.image.load(os.path.join(folder, "card_back_blue.jpg"))
                back_red = pygame.image.load(os.path.join(folder, "card_back_red.jpg"))
                self.original_card_backs[0] = back_blue
                self.original_card_backs[1] = back_red
                self.card_backs[0] = pygame.transform.scale(back_blue, (CARD_WIDTH, CARD_HEIGHT))
                self.card_backs[1] = pygame.transform.scale(back_red, (CARD_WIDTH, CARD_HEIGHT))
                print("Caricati card_back_blue.jpg e card_back_red.jpg per dorso carte")
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
                print("Creati dorsi default per le carte")
        
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
        self.game_state = {}
        self.message_queue = deque()
        self.move_queue = deque()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.public_ip = None
        self.connection_in_progress = False  # Add this line to track connection attempts
        self.connection_start_time = 0  # Add this line to track when connection started
        
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
            # Prepara info connessione da condividere in lobby
            try:
                if not isinstance(self.game_state, dict):
                    self.game_state = {}
                lobby = self.game_state.setdefault('lobby_state', {})
                conn = lobby.setdefault('connection_info', {})
                conn['local_ip'] = get_local_ip()
                conn['public_ip'] = self.public_ip
                conn['port'] = self.port
            except Exception:
                pass
            
            # Start thread to accept connections
            threading.Thread(target=self.accept_connections, daemon=True).start()
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def connect_to_server(self):
        """Connect to the host server as a client with improved connection handling"""
        self.reconnect_attempts = 0
        self.connection_start_time = time.time()  # Track when we started connecting
        self.connection_in_progress = True
        
        # Start connection in a separate thread
        threading.Thread(target=self._connect_async, daemon=True).start()
        return True  # Return immediately - connection status will be updated asynchronously
    
    def _connect_async(self):
        """Perform connection attempts in a background thread"""
        while self.reconnect_attempts < self.max_reconnect_attempts and self.connection_in_progress:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Set a shorter timeout for faster failure detection
                self.socket.settimeout(2)
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None)  # Reset timeout after connection
                self.connected = True
                
                # Add a success message to the queue
                self.message_queue.append(f"Connected to {self.host}")
                
                # Start thread to receive game state updates
                threading.Thread(target=self.receive_updates, daemon=True).start()
                self.connection_in_progress = False
                return
            except Exception as e:
                self.reconnect_attempts += 1
                self.message_queue.append(f"Connection attempt {self.reconnect_attempts} failed")
                print(f"Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} failed: {e}")
                
                # Sleep briefly before retrying
                time.sleep(1)
        
        # Connection failed after all attempts
        self.connection_in_progress = False
        self.message_queue.append(f"Failed to connect to {self.host} after {self.reconnect_attempts} attempts")
        print(f"Could not connect to server after {self.max_reconnect_attempts} attempts")

    def check_connection_timeout(self, timeout_seconds=5):
        """Check if the connection attempt has timed out"""
        if self.connection_in_progress:
            elapsed = time.time() - self.connection_start_time
            if elapsed > timeout_seconds:
                self.connection_in_progress = False
                self.message_queue.append(f"Connection attempt timed out after {int(elapsed)} seconds")
                return True
        return False
    
    def accept_connections(self):
        """Accept connections from other players (for host)"""
        # Ottieni la configurazione
        is_team_vs_ai = False
        if hasattr(self, 'game_state') and self.game_state:
            # Controlla se siamo in modalità team vs AI
            is_team_vs_ai = self.game_state.get('online_type') == 'team_vs_ai'
        
        # Numero di client attesi in base alla modalità
        online_type = None
        try:
            online_type = self.game_state.get('online_type') if isinstance(self.game_state, dict) else None
        except Exception:
            online_type = None
        if online_type == 'team_vs_ai' or online_type == 'humans_plus_ai':
            expected_clients = 1
        elif online_type == 'three_humans_one_ai':
            expected_clients = 2
        else:  # all_human o sconosciuto
            expected_clients = 3
        
        while len(self.clients) < expected_clients:
            try:
                client, addr = self.socket.accept()
                
                # Assegna player_id in base alla modalità di gioco
                if is_team_vs_ai:
                    # In modalità team vs AI, assegna SEMPRE ID 2 al client (partner umano)
                    player_id = 2
                    print(f"Team vs AI: assegnato ID 2 al partner (client)")
                else:
                    # Nelle altre modalità, assegna ID progressivi (1, 2, 3)
                    player_id = len(self.clients) + 1
                
                # Invia player ID al client
                client.sendall(pickle.dumps({"type": "player_id", "id": player_id}))
                
                self.clients.append((client, player_id))
                print(f"Player {player_id} connected from {addr}")
                
                # Start thread to handle this client
                threading.Thread(target=self.handle_client, 
                                args=(client, player_id),
                                daemon=True).start()
                
                # Add message to queue
                self.message_queue.append(f"Player {player_id} connected")
                
                # Initialize/refresh lobby state for all-human mode
                if not is_team_vs_ai:
                    try:
                        lobby = self.game_state.setdefault('lobby_state', {'players': {}, 'seats': {}})
                        players = lobby.setdefault('players', {})
                        seats = lobby.setdefault('seats', {})
                        # Ensure connection info exists and broadcast it to clients
                        conn = lobby.setdefault('connection_info', {})
                        conn.setdefault('local_ip', get_local_ip())
                        conn.setdefault('public_ip', getattr(self, 'public_ip', None))
                        conn.setdefault('port', self.port)
                        # Ensure host entry exists
                        if 0 not in players:
                            players[0] = {'name': 'Host', 'team': 0, 'ready': False}
                        seats.setdefault(0, 0)
                        # Ensure this client entry exists with default team by seat
                        if player_id not in players:
                            default_team = 0 if player_id in [0, 2] else 1
                            players[player_id] = {
                                'name': f'Player {player_id}',
                                'team': default_team,
                                'ready': False
                            }
                        # Default seat assignment for this player if free
                        if player_id not in seats.values():
                            seats.setdefault(player_id, player_id)
                        # Broadcast lobby to all connected clients
                        self.broadcast_lobby_state()
                    except Exception as e:
                        print(f"Error updating/broadcasting lobby state: {e}")
                
                # If all players connected
                if len(self.clients) == expected_clients:
                    if online_type in ('team_vs_ai', 'humans_plus_ai'):
                        self.message_queue.append("All players connected, starting game...")
                        self.broadcast_start_game()
                    else:
                        # All-human and three_humans_one_ai: wait in lobby, do not auto-start
                        self.message_queue.append("All players connected. Waiting in lobby...")
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
                elif message["type"] == "lobby_update" and self.is_host:
                    # Update lobby state with client's nickname/ready/team changes
                    lobby = self.game_state.setdefault('lobby_state', {'players': {}})
                    pdata = lobby.setdefault('players', {}).setdefault(player_id, {})
                    # Allowed fields: name, ready; team is fixed by seat (0&2 vs 1&3)
                    if 'name' in message:
                        pdata['name'] = str(message['name'])[:20]
                    if 'ready' in message:
                        pdata['ready'] = bool(message['ready'])
                    # Ensure team based on seat
                    pdata['team'] = 0 if player_id in [0, 2] else 1
                    # Broadcast updated lobby to all
                    self.broadcast_lobby_state()
                elif message.get("type") == "lobby_swap_request" and self.is_host:
                    # Client requests to swap occupants between specific seats
                    try:
                        source_seat = int(message.get('source_seat'))
                        target_seat = int(message.get('target_seat'))
                    except Exception:
                        source_seat = None
                        target_seat = None
                    lobby = self.game_state.setdefault('lobby_state', {'players': {}, 'seats': {}})
                    seats = lobby.setdefault('seats', {})
                    online_type = self.game_state.get('online_type') if isinstance(self.game_state, dict) else None
                    ai_seat = lobby.get('ai_seat') if isinstance(lobby, dict) else None
                    if target_seat is not None and source_seat is not None and target_seat in [0,1,2,3] and source_seat in [0,1,2,3] and target_seat != source_seat:
                        # Move AI seat if requested in 3v1
                        if online_type == 'three_humans_one_ai' and ai_seat == source_seat:
                            prev_ai = ai_seat
                            lobby['ai_seat'] = target_seat
                            other_pid = seats.get(target_seat, None)
                            if other_pid is None:
                                seats.pop(prev_ai, None)
                            else:
                                seats[prev_ai] = other_pid
                            seats.pop(target_seat, None)
                        else:
                            occ = seats.get(source_seat, None)
                            if occ is None:
                                return
                            other_pid = seats.get(target_seat, None)
                            seats[target_seat] = occ
                            if other_pid is None:
                                seats.pop(source_seat, None)
                            else:
                                seats[source_seat] = other_pid
                        self.broadcast_lobby_state()
                
            except Exception as e:
                print(f"Error handling client {player_id}: {e}")
                break
        
        # Client disconnected
        print(f"Player {player_id} disconnected")
        self.message_queue.append(f"Player {player_id} disconnected")
    
    def receive_updates(self):
        """Receive game state updates from server (for clients) with improved buffer handling"""
        buffer_size = 16384  # Aumentato da 8192 a 16KB per gestire stati di gioco più grandi
        
        while self.connected:
            try:
                # Utilizziamo un approccio di accumulo per gestire messaggi di grandi dimensioni
                data = bytearray()
                
                # Leggi i dati in più chunk se necessario
                while True:
                    chunk = self.socket.recv(buffer_size)
                    if not chunk:  # Connessione chiusa
                        if not data:  # Nessun dato ricevuto
                            raise ConnectionError("Connection closed by server")
                        break
                    
                    data.extend(chunk)
                    
                    # Proviamo a vedere se abbiamo ricevuto il messaggio completo
                    try:
                        # Tenta di decodificare, se ha successo abbiamo il messaggio completo
                        pickle.loads(data)
                        break  # Messaggio completo ricevuto
                    except (pickle.UnpicklingError, EOFError):
                        # Dati incompleti, continua a leggere
                        continue
                
                # Ora dovremmo avere un messaggio completo
                try:
                    message = pickle.loads(data)
                except Exception as e:
                    print(f"Errore nella deserializzazione del messaggio: {e}")
                    print(f"Ricevuti {len(data)} bytes, possibilmente dati corrotti")
                    continue  # Salta questo messaggio e continua
                
                # Process different message types
                if message["type"] == "player_id":
                    self.player_id = message["id"]
                    print(f"Assigned player ID: {self.player_id}")
                elif message["type"] == "game_state":
                    self.game_state = message["state"]
                elif message["type"] == "rules":
                    # Ricezione regole dall'host
                    if isinstance(message.get("rules"), dict):
                        if hasattr(self, 'game_state') and self.game_state is None:
                            self.game_state = {}
                        # Salva su app.game_config per uso UI
                        if hasattr(self, 'app') and hasattr(self.app, 'game_config'):
                            self.app.game_config.setdefault("rules", {})
                            self.app.game_config["rules"].update(message["rules"])
                        # Messaggio a schermo
                        self.message_queue.append("Regole partita sincronizzate dall'host")
                elif message["type"] == "lobby_state":
                    # Sync lobby state from host
                    if not isinstance(self.game_state, dict):
                        self.game_state = {}
                    self.game_state['lobby_state'] = message.get('state', {})
                    self.message_queue.append("Lobby aggiornata dall'host")
                elif message["type"] == "start_game":
                    self.message_queue.append("Game starting!")
                elif message["type"] == "player_names":
                    # Store player names into app config for GameScreen to use
                    try:
                        if hasattr(self, 'app') and hasattr(self.app, 'game_config'):
                            self.app.game_config['player_names'] = message.get('names', {})
                            self.message_queue.append("Nicknames sincronizzati")
                    except Exception:
                        pass
                elif message["type"] == "series_state":
                    # Update series and overlay state from host (store in network.game_state)
                    state = message.get('state', {})
                    # Save series state payload into network.game_state for GameScreen to consume
                    try:
                        if not isinstance(self.game_state, dict):
                            self.game_state = {}
                        self.game_state['series_state'] = state
                        self.message_queue.append("Serie sincronizzata dall'host")
                    except Exception:
                        pass
                elif message["type"] == "chat":
                    self.message_queue.append(f"Player {message['player_id']}: {message['text']}")
                
            except socket.timeout:
                # Timeout è normale, continua il ciclo
                continue
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
        """Enhanced broadcast method with more reliable state transmission"""
        if not self.is_host:
            return
            
        # Create a deep copy of the game state to avoid reference issues
        state_copy = {}
        if self.game_state:
            # Manually copy the state to ensure it's a complete deep copy
            for key, value in self.game_state.items():
                if isinstance(value, dict):
                    state_copy[key] = {}
                    for k, v in value.items():
                        # Handle nested lists (like hands)
                        if isinstance(v, list):
                            state_copy[key][k] = [item.copy() if isinstance(item, list) else item for item in v]
                        else:
                            state_copy[key][k] = v
                elif isinstance(value, list):
                    state_copy[key] = [item.copy() if isinstance(item, list) else item for item in value]
                else:
                    state_copy[key] = value
        
        # ENHANCED: Log the content of the state being sent
        hands_info = "Hands: "
        if 'hands' in state_copy:
            for player_id, hand in state_copy['hands'].items():
                hands_info += f"Player {player_id}: {len(hand)} cards, "
        
        table_info = f"Table: {state_copy.get('table', [])}"
        current_player = f"Current player: {state_copy.get('current_player', 'unknown')}"
        
        print(f"Host broadcasting state: {current_player}, {table_info}, {hands_info}")
        
        # Create the message with the full state
        message = {
            "type": "game_state", 
            "state": state_copy,
        }
        
        try:
            # Use a larger buffer for the pickle to accommodate larger game states
            # IMPROVED: Usa un protocollo pickle più efficiente (5 è il più recente)
            protocol = pickle.HIGHEST_PROTOCOL
            if hasattr(pickle, 'DEFAULT_PROTOCOL'):
                protocol = min(pickle.DEFAULT_PROTOCOL, 4)  # Usa massimo 4 per compatibilità
                
            data = pickle.dumps(message, protocol=protocol)
            data_size = len(data)
            print(f"Broadcasting game state: {data_size} bytes")
            
            for client, player_id in self.clients:
                try:
                    # IMPROVED: Invia i dati in chunk più piccoli per evitare problemi di buffer
                    chunk_size = 4096
                    for i in range(0, len(data), chunk_size):
                        chunk = data[i:i + chunk_size]
                        client.sendall(chunk)
                        
                    print(f"State sent to client (player {player_id}): {data_size} bytes")
                except Exception as e:
                    print(f"Error sending to client (player {player_id}): {e}")
        except Exception as e:
            print(f"Error serializing game state: {e}")
    
    def broadcast_start_game(self):
        """Broadcast game start signal to all clients (host only)"""
        if not self.is_host:
            return
            
        # Prima invia le regole correnti
        rules = {}
        try:
            if hasattr(self, 'game_state') and isinstance(self.game_state, dict):
                # Se l'host ha salvato regole nello stato, includile
                rules = self.game_state.get('rules', {})
        except Exception:
            pass
        if not rules:
            # Fallback: prova a prendere dalle config globali dell'app
            try:
                import builtins
                if hasattr(builtins, 'app') and hasattr(builtins.app, 'game_config'):
                    rules = builtins.app.game_config.get('rules', {})
            except Exception:
                rules = {}

        rules_msg = {"type": "rules", "rules": rules}
        data_rules = pickle.dumps(rules_msg)
        for client, _ in self.clients:
            try:
                client.sendall(data_rules)
            except:
                pass

        # Send player names if present
        try:
            names = None
            if isinstance(self.game_state, dict):
                names = self.game_state.get('player_names')
            if not names:
                import builtins
                if hasattr(builtins, 'app') and hasattr(builtins.app, 'game_config'):
                    names = builtins.app.game_config.get('player_names')
            if names:
                names_msg = {"type": "player_names", "names": names}
                data_names = pickle.dumps(names_msg)
                for client, _ in self.clients:
                    try:
                        client.sendall(data_names)
                    except:
                        pass
        except Exception:
                pass

        message = {"type": "start_game"}
        data = pickle.dumps(message)
        
        for client, _ in self.clients:
            try:
                client.sendall(data)
            except:
                pass

    def broadcast_series_state(self, state: dict):
        """Broadcast series/overlay state to clients (host only)."""
        if not self.is_host:
            return
        msg = {"type": "series_state", "state": state}
        data = pickle.dumps(msg)
        for client, _ in self.clients:
            try:
                client.sendall(data)
            except Exception:
                pass
    
    def broadcast_lobby_state(self):
        """Broadcast lobby state to clients (host only)"""
        if not self.is_host:
            return
        state = self.game_state.get('lobby_state', {})
        try:
            msg = {"type": "lobby_state", "state": state}
            data = pickle.dumps(msg)
            for client, _ in self.clients:
                try:
                    client.sendall(data)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error broadcasting lobby: {e}")

    def send_lobby_update(self, name: str = None, ready: bool = None):
        """Send a lobby update to host (for clients)."""
        if self.is_host or not self.connected or not self.socket:
            return
        payload = {"type": "lobby_update"}
        if name is not None:
            payload['name'] = name
        if ready is not None:
            payload['ready'] = bool(ready)
        try:
            self.socket.sendall(pickle.dumps(payload))
        except Exception as e:
            print(f"Error sending lobby update: {e}")
    
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

class GameOptionsScreen(BaseScreen):
    """Intermediate screen to configure game options before starting"""
    def __init__(self, app):
        super().__init__(app)
        self.title_font = pygame.font.SysFont(None, 48)
        self.info_font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 22)

        # Internal copy of rules to edit safely
        self.rules = {}

        # UI controls (simple buttons/toggles and +/-)
        self.controls = {}

        # Buttons
        self.start_button = None
        self.back_button = None
        self.reset_button = None

        # Cached layout rects
        self.sections = {}

    def enter(self):
        super().enter()
        # Clone defaults each time
        defaults = getattr(self.app.screens["mode"], "default_rules", {})
        # Preserve previously set rules if present in app.game_config
        existing = self.app.game_config.get("rules", {})
        merged = defaults.copy()
        merged.update(existing)
        self.rules = merged
        self.setup_layout()

    def calculate_section_heights(self):
        """Calculate dynamic heights for each section based on content"""
        mode = self.rules.get("mode_type", "points")

        # Duration section
        # radio buttons, one stepper (target/mani) and tiebreak
        duration_items = 3
        if mode == "hands":
            duration_items += 0  # still one stepper, just label changes
        duration_height = 56 + duration_items * 36  # tighter spacing

        # Variants section
        # Base toggles: AP, last_scopa, re_bello, napola
        variant_toggles = 4
        # AP-dependent
        if self.rules.get("asso_piglia_tutto", False):
            variant_toggles += 2  # scopa_on_asso_piglia_tutto + posabile
            if self.rules.get("asso_piglia_tutto_posabile", False):
                variant_toggles += 1  # only_empty
        # Napola scoring line if enabled
        if self.rules.get("napola", False):
            variant_toggles += 1
        # One stepper for max consecutive scope
        variant_steppers = 1
        variants_height = 56 + variant_toggles * 36 + variant_steppers * 40

        # Setup section (AI subsection rimosso)
        setup_items = 2  # starting team + last cards to dealer
        # Altezza più aderente al contenuto: padding top/bottom 8px, item ~28-30px con gap
        combined_height = 8 + setup_items * 34 + 8

        # Visibility section (only for local multiplayer and team_vs_ai)
        is_local_mode = self.app.game_config.get("mode") in ("local_multiplayer", "team_vs_ai")
        visibility_items = 1 if is_local_mode else 0
        # Altezza compatta anche qui
        visibility_height = 0 if visibility_items == 0 else (8 + visibility_items * 34 + 8)

        return {
            "duration": duration_height,
            "variants": variants_height,
            "setup_ai": combined_height,
            "visibility": visibility_height,
        }

    def setup_layout(self):
        width = self.app.window_width
        height = self.app.window_height
        # Maggior spazio tra i 4 box delle opzioni
        pad = int(height * 0.07)
        col_w = int(width * 0.42)

        # Calculate dynamic heights
        section_heights = self.calculate_section_heights()

        # Define sections with dynamic heights
        left_x = int(width * 0.06)
        right_x = int(width * 0.52)
        top_y = int(height * 0.19)
        
        # Left column - variants + visibility (if any)
        var_h = section_heights["variants"]
        vis_h = section_heights.get("visibility", 0)
        
        # Right column - duration at top, setup below
        dur_h = section_heights["duration"]
        setup_ai_h = section_heights["setup_ai"]

        self.sections = {
            "variants": pygame.Rect(left_x, top_y, col_w, var_h),
            "duration": pygame.Rect(right_x, top_y, col_w, dur_h),
            "setup_ai": pygame.Rect(right_x, top_y + dur_h + pad, col_w, setup_ai_h),
        }
        # Add visibility section only if it has height
        if vis_h > 0:
            self.sections["visibility"] = pygame.Rect(left_x, top_y + var_h + pad, col_w, vis_h)

        # Buttons at bottom - position based on tallest column
        left_col_bottom = top_y + var_h + (pad + vis_h if vis_h > 0 else 0)
        right_col_bottom = top_y + dur_h + pad + setup_ai_h
        content_bottom = max(left_col_bottom, right_col_bottom)
        
        bw = int(width * 0.22)
        bh = int(height * 0.08)
        by = min(height - bh - pad, content_bottom + pad)
        self.start_button = Button(width//2 - bw//2, by, bw, bh, "Start", DARK_GREEN, WHITE)
        self.back_button = Button(left_x, by, int(bw*0.7), bh, "Back", HIGHLIGHT_RED, WHITE)
        self.reset_button = Button(right_x + col_w - int(bw*0.7), by, int(bw*0.7), bh, "Reset", DARK_BLUE, WHITE)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.done = True
                self.next_screen = None
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                # Handle clicks on toggles and steppers
                self._handle_click_controls(pos)

                if self.start_button and self.start_button.is_clicked(pos):
                    self._on_start()
                elif self.back_button and self.back_button.is_clicked(pos):
                    self.done = True
                    self.next_screen = "mode"
                elif self.reset_button and self.reset_button.is_clicked(pos):
                    # Reset to defaults (single preset)
                    defaults = getattr(self.app.screens["mode"], "default_rules", {})
                    self.rules = defaults.copy()

    def _toggle(self, key):
        self.rules[key] = not bool(self.rules.get(key, False))

    def _step_int(self, key, delta, lo=None, hi=None):
        val = int(self.rules.get(key, 0)) + delta
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        self.rules[key] = val

    def _cycle(self, key, values):
        cur = self.rules.get(key, values[0])
        try:
            idx = values.index(cur)
        except ValueError:
            idx = 0
        self.rules[key] = values[(idx + 1) % len(values)]

    def _handle_click_controls(self, pos):
        # Duration controls
        if getattr(self, 'mode_points_rect', None) and self.mode_points_rect.collidepoint(pos):
            self.rules["mode_type"] = "points"
        if getattr(self, 'mode_hands_rect', None) and self.mode_hands_rect.collidepoint(pos):
            self.rules["mode_type"] = "hands"
        if getattr(self, 'mode_oneshot_rect', None) and self.mode_oneshot_rect.collidepoint(pos):
            self.rules["mode_type"] = "oneshot"

        if getattr(self, 'tp_minus_rect', None) and self.tp_minus_rect.collidepoint(pos):
            self._step_int("target_points", -1, lo=1, hi=200)
        if getattr(self, 'tp_plus_rect', None) and self.tp_plus_rect.collidepoint(pos):
            self._step_int("target_points", +1, lo=1, hi=200)

        if getattr(self, 'nh_minus_rect', None) and self.nh_minus_rect.collidepoint(pos):
            self._step_int("num_hands", -1, lo=1, hi=99)
        if getattr(self, 'nh_plus_rect', None) and self.nh_plus_rect.collidepoint(pos):
            self._step_int("num_hands", +1, lo=1, hi=99)

        if getattr(self, 'tb_rect', None) and self.tb_rect.collidepoint(pos):
            self._cycle("tiebreak", ["single", "+2", "allow_draw"])

        # Variants
        # Base variants always available
        base_variants = [
            ("asso_piglia_tutto", getattr(self, 'ap_rect', None)),
            ("scopa_on_last_capture", getattr(self, 'last_scopa_rect', None)),
            ("re_bello", getattr(self, 'rb_rect', None)),
            ("napola", getattr(self, 'nap_rect', None)),
        ]
        for key, rect in base_variants:
            if rect is not None and rect.collidepoint(pos):
                self._toggle(key)

        # AP-dependent variants
        if self.rules.get("asso_piglia_tutto", False):
            if getattr(self, 'ap_scopa_rect', None) and self.ap_scopa_rect.collidepoint(pos):
                self._toggle("scopa_on_asso_piglia_tutto")
            if getattr(self, 'ap_place_rect', None) and self.ap_place_rect.collidepoint(pos):
                self._toggle("asso_piglia_tutto_posabile")
            if getattr(self, 'ap_place_only_empty_rect', None) and self.ap_place_only_empty_rect.collidepoint(pos):
                self._toggle("asso_piglia_tutto_posabile_only_empty")

        if self.rules.get("napola", False) and getattr(self, 'nap_scoring_rect', None) and self.nap_scoring_rect.collidepoint(pos):
            self._cycle("napola_scoring", ["fixed3", "length"])

        if getattr(self, 'max_scope_minus_rect', None) and self.max_scope_minus_rect.collidepoint(pos):
            cur = self.rules.get("max_consecutive_scope")
            cur = 0 if cur is None else int(cur)
            cur = max(0, cur - 1)
            self.rules["max_consecutive_scope"] = None if cur == 0 else cur
        if getattr(self, 'max_scope_plus_rect', None) and self.max_scope_plus_rect.collidepoint(pos):
            cur = self.rules.get("max_consecutive_scope")
            cur = 0 if cur is None else int(cur)
            cur = min(9, cur + 1)
            self.rules["max_consecutive_scope"] = None if cur == 0 else cur

        # Setup
        if getattr(self, 'start_team_rect', None) and self.start_team_rect.collidepoint(pos):
            self._cycle("starting_team", ["random", "team0", "team1"])
        if getattr(self, 'last_cards_rect', None) and self.last_cards_rect.collidepoint(pos):
            self._toggle("last_cards_to_dealer")

        # Visibility (only if rect exists)
        if getattr(self, 'only_turn_cards_rect', None) and self.only_turn_cards_rect.collidepoint(pos):
            self._toggle("show_only_current_turn_cards")

        # (Rimosso) Tempo per mossa AI

    def _on_start(self):
        # Salva regole in config
        self.app.game_config["rules"] = self.rules.copy()
        # Se stiamo ospitando online, torna alla schermata host-mode (scelta 4 umani / 2v2)
        if self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host"):
            # Torna alla schermata mode e apri direttamente il sottomenu host
            self.done = True
            self.next_screen = "mode"
            # Imposta un flag per aprire la schermata host al rientro
            self.app.game_config["open_host_screen"] = True
        else:
            # Avvia il gioco normalmente
            self.done = True
            self.next_screen = "game"

    def update(self):
        pass

    def draw_toggle(self, surface, rect, label, active):
        pygame.draw.rect(surface, DARK_BLUE, rect, border_radius=8)
        pygame.draw.rect(surface, GOLD, rect, 2, border_radius=8)
        text = f"{label}: {'ON' if active else 'OFF'}"
        surf = self.small_font.render(text, True, WHITE)
        surface.blit(surf, surf.get_rect(center=rect.center))

    def draw_stepper(self, surface, label, value, center, w, h):
        # label (più vicino ai pulsanti)
        label_surf = self.small_font.render(f"{label}: {value}", True, WHITE)
        gap = max(6, int(h * 0.45))
        label_rect = label_surf.get_rect(midtop=(center[0], center[1] - gap))
        surface.blit(label_surf, label_rect)
        # buttons
        minus = pygame.Rect(center[0] - w, center[1], h, h)
        plus = pygame.Rect(center[0] + w - h, center[1], h, h)
        pygame.draw.rect(surface, DARK_BLUE, minus, border_radius=6)
        pygame.draw.rect(surface, DARK_BLUE, plus, border_radius=6)
        pygame.draw.rect(surface, GOLD, minus, 2, border_radius=6)
        pygame.draw.rect(surface, GOLD, plus, 2, border_radius=6)
        m_s = self.small_font.render("-", True, WHITE)
        p_s = self.small_font.render("+", True, WHITE)
        surface.blit(m_s, m_s.get_rect(center=minus.center))
        surface.blit(p_s, p_s.get_rect(center=plus.center))
        return minus, plus

    def draw_radio3(self, surface, labels, active_idx, rect):
        # Simple 3 radio buttons horizontal with highlighted active selection
        w = rect.width // 3
        rects = []
        for i, lab in enumerate(labels):
            r = pygame.Rect(rect.left + i*w, rect.top, w - 4, rect.height)
            pygame.draw.rect(surface, DARK_BLUE, r, border_radius=8)
            border_color = GOLD if i == active_idx else (120, 120, 120)
            pygame.draw.rect(surface, border_color, r, 2, border_radius=8)
            surf = self.small_font.render(lab, True, WHITE)
            surface.blit(surf, surf.get_rect(center=r.center))
            rects.append(r)
        return rects

    def draw_section_title(self, surface, rect, text):
        title = self.title_font.render(text, True, GOLD)
        surface.blit(title, title.get_rect(midtop=(rect.centerx, rect.top - 40)))

    def draw(self, surface):
        width = self.app.window_width
        height = self.app.window_height
        surface.blit(self.app.resources.background, (0, 0))

        # Recalculate layout in case mode changed (affects duration section height)
        self.setup_layout()

        # Title
        title = self.title_font.render("Opzioni di Partita", True, GOLD)
        surface.blit(title, title.get_rect(center=(width//2, int(height*0.08))))

        # Variants (left)
        var_rect = self.sections["variants"]
        pygame.draw.rect(surface, (10, 10, 40), var_rect, border_radius=10)
        pygame.draw.rect(surface, GOLD, var_rect, 2, border_radius=10)
        self.draw_section_title(surface, var_rect, "Varianti / House Rules")

        # Duration (right top)
        dur_rect = self.sections["duration"]
        pygame.draw.rect(surface, (10, 10, 40), dur_rect, border_radius=10)
        pygame.draw.rect(surface, GOLD, dur_rect, 2, border_radius=10)
        self.draw_section_title(surface, dur_rect, "Durata / Condizioni di Vittoria")

        # Radios mode (compact)
        mode = self.rules.get("mode_type", "points")
        radios = self.draw_radio3(surface, ["A punti", "A mani", "One-shot"],
                                  ["points","hands","oneshot"].index(mode),
                                  pygame.Rect(dur_rect.left+10, dur_rect.top+8, dur_rect.width-20, 34))
        self.mode_points_rect, self.mode_hands_rect, self.mode_oneshot_rect = radios

        # Compact stepper just below radios
        self.nh_minus_rect = self.nh_plus_rect = None
        self.tp_minus_rect = self.tp_plus_rect = None
        center_y = dur_rect.top + 8 + 34 + 26
        if mode == "points":
            tp_center = (dur_rect.centerx, center_y)
            self.tp_minus_rect, self.tp_plus_rect = self.draw_stepper(
                surface, "Target punti", self.rules.get("target_points", 21), tp_center,
                int(dur_rect.width*0.3), 32)
        elif mode == "hands":
            nh_center = (dur_rect.centerx, center_y)
            self.nh_minus_rect, self.nh_plus_rect = self.draw_stepper(
                surface, "Numero mani", self.rules.get("num_hands", 1), nh_center,
                int(dur_rect.width*0.3), 32)
        # Tiebreak option (compact, directly below)
        tb_rect = pygame.Rect(dur_rect.left + 10, center_y + 32 + 10, dur_rect.width - 20, 26)
        self.tb_rect = tb_rect
        tb_text = f"Spareggio: {self.rules.get('tiebreak','single')}"
        pygame.draw.rect(surface, DARK_BLUE, tb_rect, border_radius=6)
        pygame.draw.rect(surface, GOLD, tb_rect, 2, border_radius=6)
        surface.blit(self.small_font.render(tb_text, True, WHITE),
                    self.small_font.render(tb_text, True, WHITE).get_rect(center=tb_rect.center))

        # Variants content (moved above) continues

        # toggles (more compact gaps)
        h = 30
        y = var_rect.top + 8
        self.ap_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h); y += h+8
        self.draw_toggle(surface, self.ap_rect, "Asso piglia tutto", self.rules.get("asso_piglia_tutto", False))

        # Show AP-dependent options only when AP is enabled
        if self.rules.get("asso_piglia_tutto", False):
            self.ap_scopa_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h); y += h+8
            self.draw_toggle(surface, self.ap_scopa_rect, "Conta scopa con Asso piglia tutto", self.rules.get("scopa_on_asso_piglia_tutto", False))

            # Ace placeability
            self.ap_place_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h); y += h+8
            self.draw_toggle(surface, self.ap_place_rect, "Asso piglia tutto posabile", self.rules.get("asso_piglia_tutto_posabile", False))

            if self.rules.get("asso_piglia_tutto_posabile", False):
                self.ap_place_only_empty_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h); y += h+8
                self.draw_toggle(surface, self.ap_place_only_empty_rect, "Posabile solo a tavolo vuoto", self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
            else:
                self.ap_place_only_empty_rect = None
        else:
            self.ap_scopa_rect = None
            self.ap_place_rect = None
            self.ap_place_only_empty_rect = None

        self.last_scopa_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h); y += h+8
        self.draw_toggle(surface, self.last_scopa_rect, "Scopa sull'ultima presa", self.rules.get("scopa_on_last_capture", False))
        self.rb_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h); y += h+8
        self.draw_toggle(surface, self.rb_rect, "Re Bello (Re di denari)", self.rules.get("re_bello", False))
        self.nap_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h)
        self.draw_toggle(surface, self.nap_rect, "Napola (A-2-3... di denari)", self.rules.get("napola", False))
        y += h+6
        # Mostra la riga di scoring solo se Napola è attiva
        if self.rules.get("napola", False):
            self.nap_scoring_rect = pygame.Rect(var_rect.left+10, y, var_rect.width-20, h)
            ns_text = f"Punteggio Napola: {self.rules.get('napola_scoring','fixed3')}"
            self.draw_toggle(surface, self.nap_scoring_rect, ns_text, True)
            y += h+6
        else:
            self.nap_scoring_rect = None
        # max consecutive scope stepper subito dopo Napola
        cur = self.rules.get("max_consecutive_scope")
        cur_disp = 0 if cur is None else int(cur)
        self.max_scope_minus_rect, self.max_scope_plus_rect = self.draw_stepper(
            surface, "Limite scope consecutive (0=nessuno)", cur_disp,
            (var_rect.centerx, y + 20), int(var_rect.width*0.3), 30)
        y += 46

        # Sezione visibilità carte (solo locale)
        is_local_mode = self.app.game_config.get("mode") in ("local_multiplayer", "team_vs_ai")
        if is_local_mode and "visibility" in self.sections:
            vis_rect = self.sections["visibility"]
            pygame.draw.rect(surface, (10, 10, 40), vis_rect, border_radius=10)
            pygame.draw.rect(surface, GOLD, vis_rect, 2, border_radius=10)
            self.draw_section_title(surface, vis_rect, "Visibilità carte")
            vh = 30
            vy = vis_rect.top + 8
            self.only_turn_cards_rect = pygame.Rect(vis_rect.left+10, vy, vis_rect.width-20, vh)
            self.draw_toggle(surface, self.only_turn_cards_rect, "Carte scoperte solo quelle di quello di turno", self.rules.get("show_only_current_turn_cards", False))

        # Setup section (right bottom)
        setup_ai_rect = self.sections["setup_ai"]
        pygame.draw.rect(surface, (10, 10, 40), setup_ai_rect, border_radius=10)
        pygame.draw.rect(surface, GOLD, setup_ai_rect, 2, border_radius=10)
        self.draw_section_title(surface, setup_ai_rect, "Setup tavolo")
        
        # Setup content (compact)
        y_offset = setup_ai_rect.top + 8
        st_text = f"Chi inizia: {self.rules.get('starting_team','random')}"
        self.start_team_rect = pygame.Rect(setup_ai_rect.left+10, y_offset, setup_ai_rect.width-20, 28)
        self.draw_toggle(surface, self.start_team_rect, st_text, True)
        y_offset += 36
        self.last_cards_rect = pygame.Rect(setup_ai_rect.left+10, y_offset, setup_ai_rect.width-20, 28)
        self.draw_toggle(surface, self.last_cards_rect, "Ultime carte al team dell'ultima presa", self.rules.get("last_cards_to_dealer", True))
        y_offset += 36

        # (Sezione AI rimossa)

        # Bottom buttons
        self.back_button.draw(surface)
        self.start_button.draw(surface)
        self.reset_button.draw(surface)
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
        self.join_button = None
        
        # Status message
        self.status_message = ""
        
        self.host_screen_active = False
        self.online_mode_buttons = []
        self.selected_online_mode = 0  # 0: 4 Players, 1: 2v2 with AI

        # Default rules/options (used by the upcoming options screen)
        self.default_rules = {
            "mode_type": "points",          # points | hands | oneshot
            "target_points": 21,
            "num_hands": 1,
            "tiebreak": "single",          # single | +2 | allow_draw
            # Varianti
            "asso_piglia_tutto": False,
            "scopa_on_asso_piglia_tutto": False,
            "scopa_on_last_capture": False,
            "re_bello": False,
            "napola": False,
            "napola_scoring": "fixed3",
            "max_consecutive_scope": None,
            # Visibilità carte (solo locale)
            "show_only_current_turn_cards": False,
            # Nuove opzioni AP posabilità
            "asso_piglia_tutto_posabile": False,
            "asso_piglia_tutto_posabile_only_empty": False,
            # Setup
            "starting_team": "random",     # random | team0 | team1
            "last_cards_to_dealer": True,
            # AI/tempo mossa
            "move_time_ms": 0,
        }
    
    def enter(self):
        super().enter()
        # Resetta lo stato della schermata quando si rientra nel menu
        self.status_message = ""
        self.ip_input = ""
        self.ip_input_active = False
        self.host_screen_active = False
        self.loading = False
        self.waiting_for_other_player = False
        # Aggiorna l'immagine di sfondo
        self.bg_image = self.app.resources.background
        # Reimposta il layout
        self.setup_layout()
        # Se richiesto, apri direttamente la schermata host online dopo le opzioni
        if self.app.game_config.get("open_host_screen"):
            self.host_screen_active = True
            # ripristina e poi pulisci il flag
            self.app.game_config.pop("open_host_screen", None)
            # prepara i pulsanti della schermata host
            self.setup_online_choice_buttons()
    
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
        ]
        
        # Difficulty selection buttons
        diff_button_width = int(width * 0.15)
        diff_button_height = int(height * 0.05)
        
        self.difficulty_buttons = [
            Button(center_x - int(width * 0.25), button_start_y + 4 * (button_height + button_spacing), 
                diff_button_width, diff_button_height, "Easy", DARK_GREEN, WHITE, 
                font_size=int(height * 0.025)),
            Button(center_x - diff_button_width//2, button_start_y + 4 * (button_height + button_spacing), 
                diff_button_width, diff_button_height, "Medium", DARK_BLUE, WHITE, 
                font_size=int(height * 0.025)),
            Button(center_x + int(width * 0.1), button_start_y + 4 * (button_height + button_spacing), 
                diff_button_width, diff_button_height, "Hard", HIGHLIGHT_RED, WHITE, 
                font_size=int(height * 0.025)),
        ]
        
        # IP input for joining online games
        self.ip_input_rect = pygame.Rect(
            center_x - int(width * 0.225),
            button_start_y + 5 * (button_height + button_spacing),  # Change from 6 to 5
            int(width * 0.3),  # Already set to correct width
            int(height * 0.05)
        )

        # Add join button next to the IP input field
        join_button_width = int(width * 0.15)
        join_button_height = int(height * 0.05)
        self.join_button = Button(
            self.ip_input_rect.right + int(width * 0.02),
            self.ip_input_rect.top,
            join_button_width,
            join_button_height,
            "Join Game",
            DARK_BLUE,
            WHITE,
            font_size=int(height * 0.025)
        )
        
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
                            if i in (0, 1, 2, 3):  # quattro modalità
                                self.selected_online_mode = i
                                # Ora non avviamo immediatamente il gioco in nessun caso,
                                # aspettiamo che l'utente prema il pulsante Start
                            elif i == 4:  # Back
                                self.host_screen_active = False
                    
                    # Controllo clic sui pulsanti di difficoltà (solo per modalità Team vs AI)
                    if self.selected_online_mode in (1, 2, 3) and hasattr(self, 'online_difficulty_buttons'):
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
                        
                # Check if join button is clicked
                if hasattr(self, 'join_button') and self.join_button and self.join_button.is_clicked(pos):
                    self.join_online_game()
            
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
            
            # Configura la base e vai alla schermata opzioni
            self.app.game_config = {
                "mode": "single_player",
                "human_players": 1,
                "ai_players": 3,
                "difficulty": self.selected_difficulty
            }
            self.done = True
            self.next_screen = "options"
            return
            
        elif button_index == 1:
            # 2 Players (Team) - Mostra animazione di caricamento
            self.loading = True
            self.loading_animation = LoadingAnimation(self.app.window_width, self.app.window_height)
            self.loading_start_time = pygame.time.get_ticks()
            self.loading_message = "Caricamento bot AI in corso"
            
            # Configura la base e vai alla schermata opzioni
            self.app.game_config = {
                "mode": "team_vs_ai",
                "human_players": 2,
                "ai_players": 2,
                "difficulty": self.selected_difficulty
            }
            self.done = True
            self.next_screen = "options"
            return
            
        # Solo per le modalità senza bot, passa direttamente alla prossima schermata
        elif button_index == 2:
            # 4 Players (Local) -> passa alla schermata opzioni
            self.done = True
            self.next_screen = "options"
            self.app.game_config = {
                "mode": "local_multiplayer",
                "human_players": 4,
                "ai_players": 0
            }
        elif button_index == 3:
            # Host Online Game - apri prima la schermata opzioni
            self.app.game_config = {
                "mode": "online_multiplayer",
                "is_host": True,
                "player_id": 0,
                # salva un flag per ritornare qui dopo le opzioni
                "open_host_screen": True
            }
            self.done = True
            self.next_screen = "options"
        elif button_index == 4:
            # Join Online Game
            if self.ip_input:
                self.join_online_game()
            else:
                # Instead of setting a status message, just activate the input
                self.ip_input_active = True
                # No status message is needed
    
    def host_online_game(self):
        """Host an online game with internet support"""
        print("\nDEBUG HOST GAME: Creating NetworkManager with is_host=True")
        self.app.network = NetworkManager(is_host=True)
        
        # CORREZIONE: Imposta il game_state PRIMA di avviare il server
        if self.selected_online_mode == 1:  # Team vs AI
            self.app.network.game_state = {
                'online_type': 'team_vs_ai'
            }
        
        if self.app.network.start_server():
            # Salva le regole correnti (se già impostate) nello stato da sincronizzare
            try:
                if hasattr(self.app, 'game_config') and 'rules' in self.app.game_config:
                    if not isinstance(self.app.network.game_state, dict):
                        self.app.network.game_state = {}
                    self.app.network.game_state['rules'] = self.app.game_config['rules']
            except Exception:
                pass
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
                # 4 Players (All Human) -> go to lobby
                self.done = True
                self.next_screen = "lobby"
                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": True,
                    "player_id": 0,
                    "online_type": "all_human"
                }
                # Initialize basic lobby state for host
                try:
                    lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}})
                    players = lobby.setdefault('players', {})
                    players[0] = {'name': 'Host', 'team': 0, 'ready': False}
                    self.app.network.broadcast_lobby_state()
                except Exception:
                    pass
                print("DEBUG HOST GAME: Lobby avviata (all_human)")
            elif self.selected_online_mode == 1:
                # 2 vs 2 (Team vs AI) - Mostra animazione di caricamento
                self.loading = True
                self.loading_animation = LoadingAnimation(self.app.window_width, self.app.window_height)
                self.loading_start_time = pygame.time.get_ticks()
                self.loading_message = "Caricamento bot AI in corso"
                
                # CRITICAL: Make sure is_host is True and explicitly set online_type
                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": True,  # CRITICAL: Set is_host flag
                    "player_id": 0,
                    "online_type": "team_vs_ai",
                    "difficulty": self.selected_difficulty
                }
                print(f"DEBUG HOST GAME: Setting game_config with is_host=True, online_type=team_vs_ai")
                
                # Force game_config to be correctly set
                print(f"DEBUG HOST GAME: Final game_config: {self.app.game_config}")
                
                # CORREZIONE CRITICA: Imposta esplicitamente waiting_for_other_player a True
                # per far sì che venga verificata la condizione in update() che attiva setup_team_vs_ai_online()
                self.waiting_for_other_player = True
                
                # Questa riga è ora ridondante ma la mantengo per sicurezza
                if hasattr(self.app, 'network') and self.app.network:
                    self.app.network.game_state = {
                        'online_type': 'team_vs_ai'
                    }
            elif self.selected_online_mode == 2:
                # 2 vs 2 (Humans + AI compagni): due umani su squadre opposte, ognuno con un'AI compagna
                self.loading = True
                self.loading_animation = LoadingAnimation(self.app.window_width, self.app.window_height)
                self.loading_start_time = pygame.time.get_ticks()
                self.loading_message = "Caricamento bot AI in corso"

                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": True,
                    "player_id": 0,
                    "online_type": "humans_plus_ai",
                    "difficulty": self.selected_difficulty
                }
                print("DEBUG HOST GAME: Setting online_type=humans_plus_ai")
                self.waiting_for_other_player = True
                if hasattr(self.app, 'network') and self.app.network:
                    self.app.network.game_state = {
                        'online_type': 'humans_plus_ai'
                    }
            else:
                # 3 umani + 1 AI (lobby). Vai in lobby per la scelta dei posti. L'AI risulta pronta.
                self.done = True
                self.next_screen = "lobby"
                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": True,
                    "player_id": 0,
                    "online_type": "three_humans_one_ai",
                    "difficulty": self.selected_difficulty
                }
                # Inizializza lobby con AI pronta su seat libero (placeholder; sarà determinato dopo i join)
                try:
                    lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}, 'seats': {}, 'ai_ready': True})
                    players = lobby.setdefault('players', {})
                    seats = lobby.setdefault('seats', {})
                    # Host presente
                    players.setdefault(0, {'name': 'Host', 'team': 0, 'ready': False})
                    seats.setdefault(0, 0)
                    # Marca flag per AI pronta
                    lobby['ai_ready'] = True
                    # Online type nella game_state
                    self.app.network.game_state['online_type'] = 'three_humans_one_ai'
                    self.app.network.broadcast_lobby_state()
                except Exception:
                    pass
    
    def join_online_game(self):
        """Join an online game with timeout handling"""
        # Use localhost if no IP entered
        host = self.ip_input if self.ip_input else "localhost"
        self.app.network = NetworkManager(is_host=False, host=host)
        
        # Start connection attempt
        self.app.network.connect_to_server()
        
        # Show status message
        self.status_message = f"Connecting to {host}..."
    
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
                # Highlight selected button for game modes (first three are modes)
                if i < 4 and i == self.selected_online_mode:
                    pygame.draw.rect(surface, GOLD, button.rect.inflate(6, 6), 3, border_radius=8)
                button.draw(surface)
            
            # Draw instructions based on selected mode
            if self.selected_online_mode == 0:
                info_text = "You'll need 3 more human players to join"
            elif self.selected_online_mode in (1, 2):
                info_text = "You'll need 1 more human player to join (for your team)"
            elif self.selected_online_mode == 3:
                info_text = "You'll need 2 more human players to join"
            
            info_surf = self.info_font.render(info_text, True, WHITE)
            info_rect = info_surf.get_rect(center=(center_x, height - height * 0.15))
            surface.blit(info_surf, info_rect)
            
            # Draw difficulty selection if the selected mode uses AI
            if self.selected_online_mode in (1, 2, 3):
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
            else:  # 2 vs 2 (Team vs AI) o Humans+AI compagni
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
        
        # Draw IP input box with appropriate colors
        ip_color = LIGHT_BLUE if self.ip_input_active else DARK_BLUE
        pygame.draw.rect(surface, ip_color, self.ip_input_rect, border_radius=5)
        pygame.draw.rect(surface, WHITE, self.ip_input_rect, 2, border_radius=5)

        # Draw IP input text with contextual placeholder
        if not self.ip_input and self.status_message and "Please enter an IP address" in self.status_message:
            # Show error directly in the input field with error color
            ip_text = "Please enter an IP address"
            text_color = HIGHLIGHT_RED
        else:
            # Normal behavior (show IP or placeholder)
            ip_text = self.ip_input if self.ip_input else "Enter IP to join"
            text_color = WHITE

        ip_surf = self.info_font.render(ip_text, True, text_color)
        ip_rect = ip_surf.get_rect(center=self.ip_input_rect.center)
        surface.blit(ip_surf, ip_rect)
        # Blinking caret for IP input when active
        if self.ip_input_active:
            # Initialize blink state if not present (backward safety)
            if not hasattr(self, 'ip_caret_visible'):
                self.ip_caret_visible = True
                self.ip_caret_last_toggle = 0
                self.ip_caret_blink_interval_ms = 500
            # Toggle timing
            now = pygame.time.get_ticks()
            if now - getattr(self, 'ip_caret_last_toggle', 0) > getattr(self, 'ip_caret_blink_interval_ms', 500):
                self.ip_caret_visible = not getattr(self, 'ip_caret_visible', True)
                self.ip_caret_last_toggle = now
            if getattr(self, 'ip_caret_visible', True):
                caret_x = min(ip_rect.right + 2, self.ip_input_rect.right - 6)
                pygame.draw.line(surface, WHITE, (caret_x, self.ip_input_rect.top + 6), (caret_x, self.ip_input_rect.bottom - 6), 2)

        # Draw join button (ADD THIS CODE)
        if hasattr(self, 'join_button') and self.join_button:
            self.join_button.draw(surface)

        # Aggiungi istruzioni per il gioco online nella schermata di join
        if self.ip_input_active:
            help_text = [
                "Per giocare su internet:", 
                "1. L'host deve configurare port forwarding sul router (porta 5555)",
                "2. Inserisci l'IP pubblico dell'host (o l'IP locale se sei sulla stessa rete)",
                "3. Assicurati che non ci siano firewall che bloccano la porta 5555"
            ]
            
            help_y = self.ip_input_rect.bottom + 15
            for line in help_text:
                help_surf = self.small_font.render(line, True, LIGHT_GRAY)
                help_rect = help_surf.get_rect(left=self.ip_input_rect.left, top=help_y)
                surface.blit(help_surf, help_rect)
                help_y += 20
        
        # Draw status message if any
        if self.status_message and "Please enter an IP address" not in self.status_message:
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
        # Further reduce vertical spacing between mode buttons for a tighter layout
        button_spacing = int(height * 0.015)
        
        title_y = int(height * 0.18)
        # Start buttons closer to the title
        button_start_y = title_y + int(height * 0.06)
        
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
                "2 vs 2 (Humans + AI compagni)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),

            Button(center_x - button_width // 2,
                button_start_y + 3 * (button_height + button_spacing),
                button_width, button_height,
                "3 umani + 1 AI (lobby)",
                DARK_BLUE, WHITE, font_size=int(height * 0.03)),
                
            Button(center_x - button_width // 2,
                button_start_y + 4 * (button_height + button_spacing),
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
        """Update screen state with connection timeout handling"""
        # Check for network connection timeout
        if hasattr(self.app, 'network') and self.app.network:
            # Verifica se la connessione ha avuto successo anche se non è più in progress
            if self.app.network.connected and not self.done:
                print("Connessione riuscita, passaggio alla schermata di gioco")
                # If we're joining a 4-human room, go to lobby first; otherwise go to game
                go_lobby = False
                try:
                    if isinstance(self.app.network.game_state, dict):
                        go_lobby = self.app.network.game_state.get('online_type') == 'all_human' or (
                            'lobby_state' in self.app.network.game_state
                        )
                except Exception:
                    pass
                self.done = True
                # Decide screen: send clients to lobby ONLY for all-human rooms or if lobby_state already present
                try:
                    gs = self.app.network.game_state if isinstance(self.app.network.game_state, dict) else {}
                except Exception:
                    gs = {}
                is_client = not self.app.network.is_host
                all_human_flag = (gs.get('online_type') == 'all_human') if isinstance(gs, dict) else False
                has_lobby = ('lobby_state' in gs) if isinstance(gs, dict) else False
                if is_client and (all_human_flag or has_lobby):
                    self.next_screen = "lobby"
                else:
                    self.next_screen = "game"
                
                # FIX: Preserva il valore is_host dal NetworkManager
                # invece di sovrascriverlo con False
                is_host = self.app.network.is_host
                
                # Preserva anche altre impostazioni esistenti
                existing_config = self.app.game_config.copy() if hasattr(self.app, 'game_config') else {}
                
                # Aggiorna la configurazione preservando i valori originali
                self.app.game_config = {
                    "mode": "online_multiplayer",
                    "is_host": is_host,  # Usa il valore corretto
                    "player_id": self.app.network.player_id  # Usa il player_id già assegnato
                }
                
                # Ripristina altre impostazioni importanti
                for key in ['online_type', 'difficulty']:
                    if key in existing_config:
                        self.app.game_config[key] = existing_config[key]
                        
                print(f"DEBUG: Game config dopo la connessione: {self.app.game_config}")
                return
                
            # Gestione normale della connessione in corso
            if self.app.network.connection_in_progress:
                # Show connecting message with dots animation
                dots = "." * ((pygame.time.get_ticks() // 500) % 4)
                self.status_message = f"Connecting to {self.app.network.host}{dots}"
                
                # Check if connection timed out
                if self.app.network.check_connection_timeout(5):  # 5-second timeout
                    self.status_message = f"Failed to connect to {self.app.network.host}"
        
        # Original update code continues here...
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

class LobbyScreen(BaseScreen):
    """Lobby for 4-human online games: players set nickname and ready before start."""
    def __init__(self, app):
        super().__init__(app)
        self.title_font = pygame.font.SysFont(None, 52)
        self.info_font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 22)
        self.input_active = False
        self.nickname = ""
        self.ready = False
        self.start_button = None
        self.ready_button = None
        self.name_input_rect = None
        self.status_message = ""
        # Caret blink state
        self.caret_visible = True
        self.caret_last_toggle = 0
        self.caret_blink_interval_ms = 500
        # Connection info rect
        self.conn_info_rect = None
        # Leave/Cancel button
        self.cancel_button = None

    def enter(self):
        super().enter()
        # Prefill nickname from config if present
        self.nickname = self.app.game_config.get('nickname', '')
        self.ready = False
        self.status_message = ""
        self.setup_layout()
        # Focus nickname input for host to show blinking caret immediately
        self.input_active = bool(self.app.network and self.app.network.is_host)
        # Ensure host initializes lobby players
        if self.app.network and self.app.network.is_host:
            # Non pre-popolare tutti i giocatori: aggiungi solo l'host (e AI seat per 3v1)
            lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}, 'seats': {}})
            players = lobby.setdefault('players', {})
            seats = lobby.setdefault('seats', {})
            # Assicurati che l'host sia presente
            if 0 not in players:
                players[0] = {'name': 'Host', 'team': 0, 'ready': False}
            seats.setdefault(0, 0)
            # Per 3v1 assegna un posto AI di default (es. seat 3)
            online_type = self.app.network.game_state.get('online_type')
            if online_type == 'three_humans_one_ai':
                lobby.setdefault('ai_seat', 3)
                lobby['ai_ready'] = True
            self.app.network.broadcast_lobby_state()

    def setup_layout(self):
        width = self.app.window_width
        height = self.app.window_height
        btn_w = int(width * 0.22)
        btn_h = int(height * 0.07)
        center_x = width // 2
        # Input rect
        self.name_input_rect = pygame.Rect(center_x - int(width * 0.25)//2, int(height * 0.30), int(width * 0.25), int(height * 0.07))
        # Ready/Start buttons
        self.ready_button = Button(center_x - btn_w - int(width*0.02), int(height * 0.45), btn_w, btn_h, "Pronto", DARK_GREEN, WHITE)
        self.start_button = Button(center_x + int(width*0.02), int(height * 0.45), btn_w, btn_h, "Avvia partita", HIGHLIGHT_BLUE, WHITE)
        # Cancel/Exit button at bottom-left
        cancel_w = int(width * 0.2)
        cancel_h = int(height * 0.06)
        cancel_x = int(width * 0.03)
        cancel_y = height - cancel_h - int(height * 0.04)
        btn_text = "Annulla partita" if (self.app.network and self.app.network.is_host) else "Esci dalla stanza"
        self.cancel_button = Button(cancel_x, cancel_y, cancel_w, cancel_h, btn_text, HIGHLIGHT_RED, WHITE)
        # Connection info panel (top-right)
        panel_w = int(width * 0.34)
        panel_h = int(height * 0.22)
        desired_x = width - panel_w - int(width*0.02)
        # Ensure we don't overlap the nickname input: keep a small gap to its right
        min_left = self.name_input_rect.right + int(width * 0.02)
        panel_x = max(desired_x, min_left)
        self.conn_info_rect = pygame.Rect(panel_x, int(height*0.16), panel_w, panel_h)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if self.name_input_rect.collidepoint(pos):
                    self.input_active = True
                else:
                    self.input_active = False
                # Copy buttons (connection info)
                if hasattr(self, 'copy_buttons'):
                    for btn_rect, value in self.copy_buttons:
                        if btn_rect.collidepoint(pos):
                            try:
                                text_to_copy = str(value)
                                if hasattr(pygame, 'scrap'):
                                    pygame.scrap.put(pygame.SCRAP_TEXT, text_to_copy.encode('utf-8'))
                                # Fallback: try Windows clipboard via os if needed
                            except Exception:
                                pass
                            # Brief UI feedback
                            self.status_message = f"Copiato: {text_to_copy}"
                            return
                # Handle seat swap clicks (left/right arrows)
                if hasattr(self, 'seat_controls') and self.app.network:
                    # Identify clicked seat control
                    for seat, ctrls in self.seat_controls.items():
                        if (ctrls.get('enabled', True)) and (ctrls['left'].collidepoint(pos) or ctrls['right'].collidepoint(pos)):
                            # compute target seat: prev or next in ring (0..3)
                            delta = -1 if ctrls['left'].collidepoint(pos) else 1
                            target_seat = (seat + delta) % 4
                            if self.app.network.is_host:
                                # Host swaps occupant of clicked seat (human) or moves AI seat in 3v1
                                lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}, 'seats': {}})
                                seats = lobby.setdefault('seats', {})
                                online_type = self.app.network.game_state.get('online_type') if isinstance(self.app.network.game_state, dict) else None
                                ai_seat = lobby.get('ai_seat') if isinstance(lobby, dict) else None
                                if online_type == 'three_humans_one_ai' and ai_seat == seat:
                                    # Move AI seat to target; shift any target occupant to previous AI seat
                                    prev_ai = ai_seat
                                    lobby['ai_seat'] = target_seat
                                    other_pid = seats.get(target_seat, None)
                                    if other_pid is None:
                                        seats.pop(prev_ai, None)
                                    else:
                                        seats[prev_ai] = other_pid
                                    seats.pop(target_seat, None)
                                else:
                                    # Move occupant from clicked seat if any
                                    occ = seats.get(seat, None)
                                    if occ is None:
                                        return
                                    # Special case: target is AI seat in 3v1 → swap AI seat with this human
                                    if online_type == 'three_humans_one_ai' and target_seat == ai_seat:
                                        prev_ai = ai_seat
                                        lobby['ai_seat'] = seat
                                        # Move human to target (AI seat)
                                        seats[target_seat] = occ
                                        # Origin becomes AI seat (no human occupant mapping)
                                        seats.pop(seat, None)
                                    else:
                                        other_pid = seats.get(target_seat, None)
                                        seats[target_seat] = occ
                                        if other_pid is None:
                                            seats.pop(seat, None)
                                        else:
                                            seats[seat] = other_pid
                                self.app.network.broadcast_lobby_state()
                            else:
                                # Client: send swap request to host
                                try:
                                    payload = {"type": "lobby_swap_request", "source_seat": seat, "target_seat": target_seat}
                                    self.app.network.socket.sendall(pickle.dumps(payload))
                                except Exception:
                                    pass
                            return
                # Cancel/Exit room
                if self.cancel_button and self.cancel_button.is_clicked(pos):
                    # Close network and return to home
                    try:
                        if hasattr(self.app, 'network') and self.app.network:
                            self.app.network.close()
                            self.app.network = None
                    except Exception:
                        pass
                    # Reset minimal game_config state
                    self.app.game_config = {}
                    self.done = True
                    self.next_screen = "mode"
                    return
                # Ready toggle
                if self.ready_button.is_clicked(pos):
                    self.ready = not self.ready
                    # Persist nickname locally
                    if self.nickname:
                        self.app.game_config['nickname'] = self.nickname
                    # Send lobby update to host (or update local lobby if host)
                    if self.app.network:
                        if self.app.network.is_host:
                            lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}})
                            pid = 0
                            pdata = lobby['players'].setdefault(pid, {})
                            if self.nickname:
                                pdata['name'] = self.nickname[:20]
                            pdata['ready'] = bool(self.ready)
                            pdata['team'] = 0
                            self.app.network.broadcast_lobby_state()
                        else:
                            self.app.network.send_lobby_update(name=self.nickname or None, ready=self.ready)
                # Start game (host only, when all ready)
                if self.start_button.is_clicked(pos):
                    if self.app.network and self.app.network.is_host:
                        lobby = self.app.network.game_state.get('lobby_state', {})
                        players = lobby.get('players', {})
                        online_type = self.app.network.game_state.get('online_type') if isinstance(self.app.network.game_state, dict) else None
                        can_start = False
                        if online_type == 'all_human':
                            others_ready = all(players.get(pid, {}).get('ready') for pid in [1,2,3]) and all(pid in players for pid in [1,2,3])
                            can_start = others_ready
                        elif online_type == 'three_humans_one_ai':
                            # Need 2 clients connected and 3 human ready; AI considered ready by default
                            human_pids = [pid for pid, pdata in players.items() if not pdata.get('is_ai')]
                            humans_ready = sum(1 for pid in human_pids if players.get(pid, {}).get('ready')) >= 3
                            can_start = humans_ready
                        if can_start:
                            # Build player_names mapping and store
                            names = {pid: players.get(pid, {}).get('name', f'Player {pid}') for pid in [0,1,2,3]}
                            self.app.game_config['player_names'] = names
                            self.app.network.game_state['player_names'] = names
                            # Switch to game and broadcast start
                            self.done = True
                            self.next_screen = "game"
                            # Ensure game mode flags
                            self.app.game_config.update({
                                "mode": "online_multiplayer",
                                "is_host": True,
                                "player_id": 0,
                                "online_type": online_type or "all_human"
                            })
                            self.app.network.broadcast_start_game()
                        else:
                            self.status_message = "Condizioni di readiness non soddisfatte."
                    else:
                        self.status_message = "Solo l'host può avviare."
            elif event.type == pygame.KEYDOWN and self.input_active:
                if event.key == pygame.K_RETURN:
                    # Toggle ready when pressing Enter
                    self.ready = not self.ready
                    if self.app.network:
                        if self.app.network.is_host:
                            lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}})
                            pdata = lobby['players'].setdefault(0, {})
                            if self.nickname:
                                pdata['name'] = self.nickname[:20]
                            pdata['ready'] = bool(self.ready)
                            pdata['team'] = 0
                            self.app.network.broadcast_lobby_state()
                        else:
                            self.app.network.send_lobby_update(name=self.nickname or None, ready=self.ready)
                elif event.key == pygame.K_BACKSPACE:
                    self.nickname = self.nickname[:-1]
                    # Live-update nickname to others
                    if self.app.network:
                        if self.app.network.is_host:
                            lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}})
                            pdata = lobby['players'].setdefault(0, {})
                            pdata['name'] = self.nickname[:20] if self.nickname else f'Player 0'
                            pdata['team'] = 0
                            self.app.network.broadcast_lobby_state()
                        else:
                            self.app.network.send_lobby_update(name=self.nickname or "")
                else:
                    if len(self.nickname) < 20:
                        self.nickname += event.unicode
                        # Live-update nickname to others
                        if self.app.network:
                            if self.app.network.is_host:
                                lobby = self.app.network.game_state.setdefault('lobby_state', {'players': {}})
                                pdata = lobby['players'].setdefault(0, {})
                                pdata['name'] = self.nickname[:20]
                                pdata['team'] = 0
                                self.app.network.broadcast_lobby_state()
                            else:
                                self.app.network.send_lobby_update(name=self.nickname)

    def update(self):
        # Toggle caret visibility
        now = pygame.time.get_ticks()
        if now - self.caret_last_toggle > self.caret_blink_interval_ms:
            self.caret_visible = not self.caret_visible
            self.caret_last_toggle = now

    def draw(self, surface):
        surface.blit(self.app.resources.background, (0, 0))
        width = self.app.window_width
        height = self.app.window_height
        center_x = width // 2
        # Title
        title = self.title_font.render("Lobby (4 giocatori)", True, WHITE)
        surface.blit(title, title.get_rect(center=(center_x, int(height*0.12))))
        # Client waiting badge
        if self.app.network and not self.app.network.is_host:
            badge_text = "In attesa dell'host…"
            badge_surf = self.small_font.render(badge_text, True, WHITE)
            padding_x = 12
            padding_y = 6
            badge_rect = badge_surf.get_rect()
            badge_bg = pygame.Rect(0, 0, badge_rect.width + 2*padding_x, badge_rect.height + 2*padding_y)
            # Place under the title
            badge_bg.center = (center_x, int(height*0.18))
            pygame.draw.rect(surface, (30, 30, 70), badge_bg, border_radius=12)
            pygame.draw.rect(surface, GOLD, badge_bg, 2, border_radius=12)
            surface.blit(badge_surf, badge_surf.get_rect(center=badge_bg.center))
        # Input label and box
        label = self.info_font.render("Nickname:", True, WHITE)
        surface.blit(label, label.get_rect(midleft=(self.name_input_rect.left, self.name_input_rect.top - 10)))
        pygame.draw.rect(surface, (20,20,50), self.name_input_rect, border_radius=8)
        pygame.draw.rect(surface, GOLD, self.name_input_rect, 2, border_radius=8)
        shown = self.nickname if self.nickname else "Inserisci nickname..."
        color = WHITE if self.nickname else LIGHT_GRAY
        txt = self.info_font.render(shown, True, color)
        surface.blit(txt, txt.get_rect(left=self.name_input_rect.left+8, centery=self.name_input_rect.centery))
        # Blinking caret when focused
        if self.input_active and self.caret_visible:
            caret_x = self.name_input_rect.left + 8 + txt.get_width() + 2
            caret_x = min(caret_x, self.name_input_rect.right - 6)
            pygame.draw.line(surface, WHITE, (caret_x, self.name_input_rect.top + 8), (caret_x, self.name_input_rect.bottom - 8), 2)
        # Players grid (2x2 by seat)
        grid_top = int(height*0.58)
        cell_w = int(width*0.22)
        cell_h = int(height*0.12)
        gap = int(width*0.02)
        lobby = {}
        if self.app.network and isinstance(self.app.network.game_state, dict):
            lobby = self.app.network.game_state.get('lobby_state', {})
        players = lobby.get('players', {})
        seats = lobby.get('seats', {})
        online_type = (self.app.network.game_state.get('online_type') if self.app.network and isinstance(self.app.network.game_state, dict) else None)
        ai_seat = lobby.get('ai_seat') if isinstance(lobby, dict) else None
        # Compute connected humans (host + connected clients)
        connected_humans = {0}
        try:
            if self.app.network and getattr(self.app.network, 'clients', None):
                for _, pid in self.app.network.clients:
                    connected_humans.add(pid)
        except Exception:
            pass
        # Seat layout: 0,1 top row; 2,3 bottom row (or keep 0..3 order)
        seat_order = [0,1,2,3]
        start_x = center_x - (cell_w*2 + gap)//2
        start_y = grid_top
        for idx, seat in enumerate(seat_order):
            col = idx % 2
            row = idx // 2
            rect = pygame.Rect(start_x + col*(cell_w+gap), start_y + row*(cell_h+gap), cell_w, cell_h)
            # Determine seated player id for this seat (None means libero)
            pid = seats.get(seat) if seat in seats else None
            team = 0 if seat in [0,2] else 1
            bg = (10,40,10) if team == 0 else (40,10,10)
            pygame.draw.rect(surface, bg, rect, border_radius=10)
            # Highlight border: thicker and cyan if this seat belongs to local player
            local_pid = getattr(self.app.network, 'player_id', None)
            border_color = GOLD
            border_width = 2
            if pid == local_pid:
                border_color = LIGHT_BLUE
                border_width = 4
            pygame.draw.rect(surface, border_color, rect, border_width, border_radius=10)
            # Seat content: AI, Human, or Libero
            pname = None
            pready = False
            is_ai_seat = False
            if online_type == 'three_humans_one_ai' and seat == ai_seat:
                # Explicit AI seat
                pname = f"AI {seat}"
                pready = True
                is_ai_seat = True
            elif pid is None:
                # Libero
                pname = "Libero"
                pready = False
            else:
                # Human or preset player info
                pname = players.get(pid, {}).get('name', f'Player {pid}')
                pready = players.get(pid, {}).get('ready', False)

            name_s = self.small_font.render(f"{pname} (Team {team})", True, WHITE if pname != "Libero" else LIGHT_GRAY)
            ready_s = self.small_font.render("Pronto" if pready else ("Libero" if pname == "Libero" else "In attesa"), True, (LIGHT_GREEN if pready else (LIGHT_GRAY if pname == "Libero" else ORANGE)))
            surface.blit(name_s, name_s.get_rect(left=rect.left+8, top=rect.top+8))
            # Place readiness text slightly lower and indented to avoid arrow overlap
            ready_rect = ready_s.get_rect()
            ready_rect.left = rect.left + 8
            ready_rect.top = rect.top + 40
            # Ensure it doesn't collide horizontally with left arrow; push right if needed
            left_rect_preview = pygame.Rect(rect.left + 6, rect.centery - 12, 24, 24)
            if ready_rect.colliderect(left_rect_preview):
                ready_rect.left = left_rect_preview.right + 6
            surface.blit(ready_s, ready_rect)
            # Draw AI badge for 3v1 on AI seat
            if online_type == 'three_humans_one_ai' and is_ai_seat:
                badge_text = "AI"
                badge_surf = self.small_font.render(badge_text, True, WHITE)
                pad_x = 8
                pad_y = 4
                brect = badge_surf.get_rect()
                badge_bg = pygame.Rect(0, 0, brect.width + 2*pad_x, brect.height + 2*pad_y)
                badge_bg.topright = (rect.right - 8, rect.top + 8)
                pygame.draw.rect(surface, (20, 70, 20), badge_bg, border_radius=10)
                pygame.draw.rect(surface, LIGHT_GREEN, badge_bg, 2, border_radius=10)
                surface.blit(badge_surf, badge_surf.get_rect(center=badge_bg.center))
            # Draw switch seat arrows (left/right) for each seat (disable on AI seat or libero)
            arrow_w = 24
            arrow_h = 24
            left_rect = pygame.Rect(rect.left + 6, rect.bottom - arrow_h - 6, arrow_w, arrow_h)
            right_rect = pygame.Rect(rect.right - arrow_w - 6, rect.bottom - arrow_h - 6, arrow_w, arrow_h)
            arrows_enabled = ((pid is not None) or (online_type == 'three_humans_one_ai' and seat == ai_seat))
            if arrows_enabled:
                pygame.draw.polygon(surface, WHITE, [(left_rect.right, left_rect.top), (left_rect.left, left_rect.centery), (left_rect.right, left_rect.bottom)])
                pygame.draw.polygon(surface, WHITE, [(right_rect.left, right_rect.top), (right_rect.right, right_rect.centery), (right_rect.left, right_rect.bottom)])
            # Save for click handling
            if not hasattr(self, 'seat_controls'):
                self.seat_controls = {}
            self.seat_controls[seat] = {'rect': rect, 'left': left_rect, 'right': right_rect, 'enabled': arrows_enabled}
        # Buttons
        self.ready_button.draw(surface)
        # Dim the start button if not host
        if self.app.network and not self.app.network.is_host:
            # Draw disabled style
            disabled = Button(self.start_button.rect.left, self.start_button.rect.top, self.start_button.rect.width, self.start_button.rect.height, self.start_button.text, (80,80,100), LIGHT_GRAY)
            disabled.draw(surface)
        else:
            self.start_button.draw(surface)
        # Cancel/Exit button
        if self.cancel_button:
            self.cancel_button.draw(surface)
        
        # Connection info panel (visible to both host and clients)
        pygame.draw.rect(surface, (15,15,35), self.conn_info_rect, border_radius=10)
        pygame.draw.rect(surface, GOLD, self.conn_info_rect, 2, border_radius=10)
        header = self.info_font.render("Info Connessione (da condividere)", True, WHITE)
        surface.blit(header, header.get_rect(midtop=(self.conn_info_rect.centerx, self.conn_info_rect.top + 8)))
        y = self.conn_info_rect.top + 38
        line_h = 24
        # Preferisci i dati broadcastati dall'host in lobby_state
        local_ip = "-"
        public_ip = "-"
        port_text = "5555"
        lobby = {}
        if self.app.network and isinstance(self.app.network.game_state, dict):
            lobby = self.app.network.game_state.get('lobby_state', {})
        conn_info = lobby.get('connection_info', {}) if isinstance(lobby, dict) else {}
        if conn_info:
            local_ip = str(conn_info.get('local_ip', local_ip))
            public_ip = str(conn_info.get('public_ip', public_ip))
            port_val = conn_info.get('port', None)
            if port_val is not None:
                port_text = str(port_val)
        else:
            # Fallbacks
            try:
                local_ip = get_local_ip()
            except Exception:
                local_ip = "-"
            if hasattr(self.app, 'network') and self.app.network:
                public_ip = getattr(self.app.network, 'public_ip', public_ip)
                port_text = str(getattr(self.app.network, 'port', 5555))
        
        lines = [
            ("IP Locale (LAN)", local_ip),
            ("IP Pubblico (Internet)", public_ip),
            ("Porta", port_text),
        ]
        # Show connections count (both see the same type of info)
        if self.app.network:
            try:
                num_clients = len(self.app.network.clients)
                lines.append(("Giocatori connessi", f"{num_clients}/3"))
            except Exception:
                pass
        # Draw each line with a copy button
        self.copy_buttons = []
        for label, value in lines:
            text = f"{label}: {value}"
            surf = self.small_font.render(text, True, LIGHT_GRAY)
            text_rect = surf.get_rect(left=self.conn_info_rect.left + 12, top=y)
            surface.blit(surf, text_rect)
            # Draw copy button (small square with '⧉')
            btn_size = int(self.small_font.get_height() * 1.2)
            btn_rect = pygame.Rect(self.conn_info_rect.right - btn_size - 10, y - 2, btn_size, btn_size)
            pygame.draw.rect(surface, (45,45,80), btn_rect, border_radius=6)
            pygame.draw.rect(surface, GOLD, btn_rect, 1, border_radius=6)
            icon_surf = self.small_font.render("⧉", True, WHITE)
            surface.blit(icon_surf, icon_surf.get_rect(center=btn_rect.center))
            # Store for click handling
            self.copy_buttons.append((btn_rect, value))
            y += line_h
        # Status
        if self.status_message:
            s = self.small_font.render(self.status_message, True, HIGHLIGHT_RED)
            surface.blit(s, s.get_rect(center=(center_x, int(height*0.9))))
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
        
        self.new_game_button = Button(20, 20, 140, 50, "Exit", 
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
        
        # Add connection loss tracking variables
        self.connection_lost = False
        self.reconnect_start_time = 0
        self.exit_button = None
        self.reconnection_attempts = 0
        self.max_reconnection_time = 60  # 60 seconds (1 minute) timeout
        
        # Replay functionality
        self.replay_button = None
        self.replay_active = False
        self.replay_moves = []
        self.replay_current_index = 0
        self.replay_animations = []
        self.replay_table_state = []
        # Ensure motion-tracking sets are fresh per match
        if hasattr(self, 'cards_in_motion'):
            self.cards_in_motion.clear()
        else:
            self.cards_in_motion = set()
        if hasattr(self, 'replay_cards_in_motion'):
            self.replay_cards_in_motion.clear()
        else:
            self.replay_cards_in_motion = set()
        # Cancel any leftover staged animation state from previous match
        self.waiting_for_animation = False
        self.pending_action = None
        
        # Message log interaction state
        self.message_minimized = False
        self.message_dragging = False
        self.message_resizing = False
        self.message_drag_offset = (0, 0)
        self.message_resize_start_mouse = (0, 0)
        self.message_resize_start_rect = None
        self.message_resize_zone_size = 14
        self.message_header_height = 28
        self.message_minimize_rect = None
        self.message_resize_rect = None
        self.message_prev_size = None
        self.message_prev_height = None
        self.message_focused = False
        # Scrollbar state
        self.scrollbar_dragging = False
        self.scrollbar_drag_offset = 0
        self.scrollbar_up_rect = None
        self.scrollbar_down_rect = None
        self.scrollbar_track_rect = None
        self.scrollbar_thumb_rect = None
        self.scrollbar_thumb_height = 0
        self.scrollbar_max_offset = 0

        # Match/series configuration and tracking
        self.series_mode = "oneshot"          # "points" | "hands" | "oneshot"
        self.series_target_points = 21
        self.series_num_hands = 1
        self.series_tiebreak = "single"       # "single" | "+2" | "allow_draw"
        self.series_scores = [0, 0]
        self.series_hands_played = 0
        self.series_prev_starter = None
        # Storico punteggi per mano (solo per modalità a punti)
        self.points_history = []  # list[(team0_points:int, team1_points:int)]
        # Conteggio mani vinte (modalità a mani)
        self.hands_won = [0, 0]
        # Recap intermedio tra le mani (modalità a punti)
        self.show_intermediate_recap = False
        self.last_hand_breakdown = None
        self._pending_next_starter = None
        self.next_hand_button = None
    
    def create_exit_button(self):
        """Create a prominent exit button when connection is lost"""
        width = self.app.window_width
        height = self.app.window_height
        
        button_width = int(width * 0.2)
        button_height = int(height * 0.06)
        
        self.exit_button = Button(
            width // 2 - button_width // 2,
            height * 0.1,
            button_width,
            button_height,
            "Exit Game",
            HIGHLIGHT_RED,
            WHITE,
            font_size=int(height * 0.03)
        )
    
    def enter(self):
        """Called when entering this screen"""
        super().enter()
        # Reset transient state so a new match doesn't inherit previous game_over and UI flags
        self.game_over = False
        self.final_breakdown = None
        self.status_message = ""
        self.selected_hand_card = None
        self.selected_table_cards = set()
        self.animations = []
        self.waiting_for_other_player = False
        self.ai_thinking = False
        self.ai_move_timer = 0
        self.replay_active = False
        self.replay_moves = []
        self.replay_current_index = 0
        self.replay_animations = []
        self.replay_table_state = []
        self.connection_lost = False
        self.exit_button = None
        self.reconnect_start_time = 0
        self.reconnection_attempts = 0
        self.messages = []
        self.message_scroll_offset = 0
        self.message_minimized = False
        self.message_dragging = False
        self.message_resizing = False
        self.message_focused = False
        self.message_prev_size = None
        self.message_prev_height = None
        self.visually_hidden_cards = {}
        self.ai_controllers = {}
        
        # Initialize game based on config
        self.initialize_game()
        
        # Initialize series settings from options
        rules = self.app.game_config.get("rules", {})
        self.series_mode = rules.get("mode_type", "oneshot")
        self.series_target_points = int(rules.get("target_points", 21))
        self.series_num_hands = int(rules.get("num_hands", 1))
        self.series_tiebreak = rules.get("tiebreak", "single")
        self.series_scores = [0, 0]
        self.series_hands_played = 0
        self.series_prev_starter = None
        self.points_history = []
        self.hands_won = [0, 0]
        self.show_intermediate_recap = False
        self.last_hand_breakdown = None
        self._pending_next_starter = None
        self.next_hand_button = None
        
        # Set up player info
        self.setup_players()
        
        # Set up layout
        self.setup_layout()
        
        # Clear card angles and other state
        self.game_over_button_rect = None
        
        # IMPORTANTE: Flag per la sincronizzazione iniziale
        self.initial_sync_done = False
        
    def exit(self):
        """Called when exiting this screen"""
        # Chiudi la connessione di rete quando si esce dalla schermata di gioco
        if hasattr(self.app, 'network') and self.app.network:
            print("Chiusura connessione di rete in corso...")
            self.app.network.close()
            self.app.network = None  # Rimuovi completamente l'oggetto network
            print("Connessione di rete terminata con successo")
    
    def initialize_game(self):
        """Initialize game environment and state with randomized starting player"""
        config = self.app.game_config
        mode = config.get("mode", "single_player")
        
        # Set difficulty
        self.ai_difficulty = config.get("difficulty", 2)  # Default: Hard
        
        # Create game environment con regole/varianti
        rules = config.get("rules", {})
        self.env = ScoponeEnvMA(rules=rules)
        
        # Determina se siamo l'host o il client in modalità online
        is_online = mode == "online_multiplayer"
        is_host = config.get("is_host", False)
        
        if is_online and not is_host:
            # Per i client online, inizializza con uno stato vuoto
            # Aspetteremo che l'host invii lo stato completo del gioco
            self.env.reset()
            print("Client: inizializzazione base, in attesa dello stato dal server")
        else:
            # Per l'host (o per gioco locale/singolo), inizializza con un giocatore iniziale casuale
            random_starter = random.randint(0, 3)
            # Rispetta scelta di chi inizia
            st = rules.get("starting_team", "random")
            if st == "team0":
                random_starter = 0
            elif st == "team1":
                random_starter = 1
            print(f"Host/locale: inizializzazione con giocatore random {random_starter}")
            self.env.reset(starting_player=random_starter)
        
        # Set up AI controllers if needed
        if mode in ["single_player", "team_vs_ai"]:
            self.setup_ai_controllers()
        
        # Set up player ID (for online games)
        if mode == "online_multiplayer":
            self.local_player_id = config.get("player_id", 0)

        # Prepara struttura serie per host
        if is_online and is_host:
            # Mantieni regole nello stato di rete per broadcast
            if hasattr(self.app, 'network') and self.app.network:
                if not isinstance(self.app.network.game_state, dict):
                    self.app.network.game_state = {}
                self.app.network.game_state['rules'] = rules
                online_type = config.get("online_type")
                if online_type:
                    self.app.network.game_state['online_type'] = online_type

    def _handle_hand_end(self, final_breakdown):
        """Handle end of a single hand and possibly continue series/match."""
        # Update series counters
        self.series_hands_played += 1
        if isinstance(final_breakdown, dict):
            hand_p0 = int(final_breakdown.get(0, {}).get("total", 0))
            hand_p1 = int(final_breakdown.get(1, {}).get("total", 0))
            # Punti cumulativi (utili per riepilogo anche in modalità a mani)
            self.series_scores[0] += hand_p0
            self.series_scores[1] += hand_p1
            # Storico punteggi per mano (sempre, per recap)
            self.points_history.append((hand_p0, hand_p1))
            # Conteggio mani vinte per modalità a mani
            if self.series_mode == "hands":
                if hand_p0 > hand_p1:
                    self.hands_won[0] += 1
                elif hand_p1 > hand_p0:
                    self.hands_won[1] += 1
                else:
                    # pareggio mano: nessuno incrementa (gestito da tiebreak in serie)
                    pass

        proceed_new_hand = False
        # Decide if series continues
        if self.series_mode == "oneshot":
            proceed_new_hand = False
        elif self.series_mode == "hands":
            if self.series_hands_played < self.series_num_hands:
                proceed_new_hand = True
            # Early termination: chiudi appena una squadra non può più superare l'altra
            remaining = max(0, self.series_num_hands - self.series_hands_played)
            # Se il leader ha un vantaggio maggiore del numero di mani restanti, termina
            lead = max(self.hands_won)
            trail = min(self.hands_won)
            if lead - trail > remaining:
                proceed_new_hand = False
        elif self.series_mode == "points":
            if max(self.series_scores) < self.series_target_points:
                proceed_new_hand = True
            else:
                # Tiebreak logic (simplified for now)
                if self.series_scores[0] == self.series_scores[1]:
                    if self.series_tiebreak == "+2":
                        # Need a 2-point lead
                        if abs(self.series_scores[0] - self.series_scores[1]) < 2:
                            proceed_new_hand = True
                    elif self.series_tiebreak == "allow_draw":
                        proceed_new_hand = False
                    else:  # "single"
                        proceed_new_hand = True

        if proceed_new_hand:
            # In modalità a punti o a mani mostra prima un recap dell'ultima mano
            if self.series_mode in ("points", "hands"):
                self.show_intermediate_recap = True
                self.last_hand_breakdown = final_breakdown
                # Calcola in anticipo il prossimo starter ma non resettare ancora
                next_starter = (self.series_prev_starter + 1) % 4 if self.series_prev_starter is not None else random.randint(0, 3)
                self._pending_next_starter = next_starter
                # In online: host broadcasta lo stato della serie + recap
                if self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host"):
                    self._broadcast_series_state()
                return
            else:
                # Start a new hand immediately (modalità a mani)
                next_starter = (self.series_prev_starter + 1) % 4 if self.series_prev_starter is not None else random.randint(0, 3)
                self.series_prev_starter = next_starter
                self.game_over = False
                self.final_breakdown = None
                self.status_message = ""
                self.selected_hand_card = None
                self.selected_table_cards = set()
                self.animations = []
                # Recreate environment with same rules
                rules = self.app.game_config.get("rules", {})
                self.env = ScoponeEnvMA(rules=rules)
                self.env.reset(starting_player=next_starter)
                # Keep AI controllers
                self.setup_players()
                self.setup_layout()
        else:
            # Series ended; show final scoreboard by setting final_breakdown to last hand (already set)
            pass

    
    def setup_ai_controllers(self):
        """Set up AI controllers based on game mode"""
        config = self.app.game_config
        mode = config.get("mode", "single_player")
        
        if mode == "single_player":
            # Crea un solo agente per squadra (come nell'addestramento)
            # Team 0: giocatori 0, 2
            # Team 1: giocatori 1, 3
            
            # Crea l'agente per la squadra 0 (giocatori 0 e 2)
            team0_agent = DQNAgent(team_id=0)
            checkpoint_path_0 = "scopone_checkpoint_team0.pth"
            if os.path.exists(checkpoint_path_0):
                team0_agent.load_checkpoint(checkpoint_path_0)
            
            # Crea l'agente per la squadra 1 (giocatori 1 e 3)
            team1_agent = DQNAgent(team_id=1)
            checkpoint_path_1 = "scopone_checkpoint_team1.pth"
            if os.path.exists(checkpoint_path_1):
                team1_agent.load_checkpoint(checkpoint_path_1)
            
            # Imposta epsilon in base alla difficoltà
            if self.ai_difficulty == 0:  # Easy
                team0_agent.epsilon = 0.3
                team1_agent.epsilon = 0.3
            elif self.ai_difficulty == 1:  # Medium
                team0_agent.epsilon = 0.1
                team1_agent.epsilon = 0.1
            else:  # Hard
                team0_agent.epsilon = 0
                team1_agent.epsilon = 0
            
            # Assegna gli agenti ai rispettivi giocatori
            # Il giocatore 0 è umano, quindi assegna agenti solo ai bot
            self.ai_controllers[1] = team1_agent  # Giocatore 1 -> Team 1
            self.ai_controllers[2] = team0_agent  # Giocatore 2 -> Team 0
            self.ai_controllers[3] = team1_agent  # Giocatore 3 -> Team 1
        
        elif mode == "team_vs_ai":
            # Players 0 and 2 are human team, 1 and 3 are AI team
            # Crea UN SOLO agente per la squadra AI (team 1)
            ai_agent = DQNAgent(team_id=1)
            
            # Carica checkpoint se disponibile
            checkpoint_path = "scopone_checkpoint_team1.pth"
            if os.path.exists(checkpoint_path):
                ai_agent.load_checkpoint(checkpoint_path)
            
            # Imposta epsilon in base alla difficoltà
            if self.ai_difficulty == 0:  # Easy
                ai_agent.epsilon = 0.3
            elif self.ai_difficulty == 1:  # Medium
                ai_agent.epsilon = 0.1
            else:  # Hard
                ai_agent.epsilon = 0
            
            # Assegna lo STESSO agente ad entrambi i giocatori AI
            self.ai_controllers[1] = ai_agent
            self.ai_controllers[3] = ai_agent
    
    def setup_players(self):
        """Set up player info objects with improved perspective handling"""
        config = self.app.game_config
        mode = config.get("mode", "single_player")
        
        # FIX: Verifica più robusta dello stato host/client
        is_host = config.get("is_host", False)
        # Controlla anche il network manager come backup
        if hasattr(self, 'app') and hasattr(self.app, 'network') and self.app.network:
            is_network_host = self.app.network.is_host
            # Stampa di debug per verificare entrambi i valori
            print(f"DEBUG HOST CHECK: config is_host={is_host}, network is_host={is_network_host}")
            # Se c'è una discrepanza, usa il valore dal network manager
            if is_host != is_network_host:
                print(f"ATTENZIONE: Discrepanza tra config.is_host e network.is_host! Uso network.is_host={is_network_host}")
                is_host = is_network_host
        
        # Reset player info
        self.players = []
        
        # Determina se siamo in modalità team vs AI online
        is_online_team_vs_ai = (mode == "online_multiplayer" and 
                            config.get("online_type") == "team_vs_ai")
        is_online_humans_plus_ai = (mode == "online_multiplayer" and 
                            config.get("online_type") == "humans_plus_ai")
        is_online_three_humans_one_ai = (mode == "online_multiplayer" and 
                            config.get("online_type") == "three_humans_one_ai")
        
        # CORREZIONE: Usa is_host invece di config.get("is_host", False)
        ai_players = []
        if not is_host and self.app.network and self.app.network.game_state:
            ai_players = self.app.network.game_state.get('ai_players', [])
            print(f"DEBUG: Sei considerato client e gli AI players from game state: {ai_players}")
        elif is_online_team_vs_ai:
            # Se siamo host, sappiamo già quali sono le AI
            ai_players = [1, 3]
            print(f"DEBUG: Sei considerato host e gli AI players from config: {config.get('ai_players', [])}")
        
        # Imposta i giocatori
        for player_id in range(4):
            # Determina il team_id
            # Le squadre sono sempre: [0, 2] = team 0, [1, 3] = team 1
            team_id = 0 if player_id in [0, 2] else 1
            
            # Determina se il giocatore è umano o AI
            if mode == "single_player":
                is_human = (player_id == 0)
                is_ai = not is_human
            elif mode == "team_vs_ai":
                is_human = player_id in [0, 2]
                is_ai = not is_human
            elif mode == "local_multiplayer":
                is_human = True
                is_ai = False
            elif mode == "online_multiplayer":
                # Per la modalità online, determina in base al tipo
                if is_online_team_vs_ai:
                    is_human = player_id not in ai_players
                    is_ai = player_id in ai_players
                elif is_online_humans_plus_ai:
                    # In questa modalità, gli AI sono sempre 2 e 3
                    is_human = player_id in [0, 1]
                    is_ai = player_id in [2, 3]
                elif is_online_three_humans_one_ai:
                    # AI è il seat non occupato da 3 umani; per default trattiamo AI come seat 3 finché la lobby non decide
                    # Per sicurezza: considera umani su quelli presenti in player_names (se arrivano da lobby)
                    human_ids = [pid for pid in range(4) if (config.get('player_names', {}).get(pid) not in (None, f'Player {pid}'))]
                    if len(human_ids) == 3:
                        is_human = player_id in human_ids
                        is_ai = not is_human
                    else:
                        # Fallback: host (0) e due client (1,2) umani, seat 3 AI
                        is_human = player_id in [0, 1, 2]
                        is_ai = player_id == 3
                else:  # All human
                    is_human = True
                    is_ai = False
            
            # Determina il nome del giocatore
            player_names = config.get('player_names', {}) if mode == "online_multiplayer" else {}
            name = None
            if is_human:
                name = player_names.get(player_id)
                if not name:
                    if player_id == self.local_player_id:
                        name = "You"
                    elif is_online_team_vs_ai and player_id in [0, 2] and player_id != self.local_player_id:
                        name = "Partner"
                    elif mode == "team_vs_ai" and player_id == 2:
                        name = "Partner"
                    else:
                        name = f"Player {player_id}"
            else:  # AI
                name = f"AI {player_id}"
            
            # Crea l'oggetto PlayerInfo
            player = PlayerInfo(
                player_id=player_id,
                name=name,
                team_id=team_id,
                is_human=is_human,
                is_ai=is_ai
            )
            
            # Aggiungi alla lista
            self.players.append(player)
        
        # Configura la prospettiva visuale per il multiplayer online
        if mode == "online_multiplayer":
            self.setup_player_perspective()
        
        # Debug: stampa informazioni sui giocatori
        for player in self.players:
            print(f"Player {player.player_id}: {player.name}, Team {player.team_id}, AI: {player.is_ai}")

    def setup_player_perspective(self):
        """Set up visualization for different client perspectives"""
        if not hasattr(self, 'visual_positions'):
            self.visual_positions = [0, 1, 2, 3]  # Default mapping
        
        # Check if local_player_id is None (not yet set by server)
        if self.local_player_id is None:
            print("Warning: local_player_id is None, using default perspective mapping")
            return  # Keep default mapping until player ID is assigned
        
        # IMPORTANTE: Stampa debug prima del calcolo
        print(f"CALCOLO POSIZIONI VISUALI PER GIOCATORE {self.local_player_id}")
        
        # Ogni giocatore vede se stesso in basso (posizione 0)
        rotation_steps = self.local_player_id
        self.visual_positions = [(i - rotation_steps) % 4 for i in range(4)]
        
        # CRUCIALE: Verifica per il giocatore 2 in modalità team vs AI
        if (self.local_player_id == 2 and 
            self.app.game_config.get("mode") == "online_multiplayer" and 
            self.app.game_config.get("online_type") == "team_vs_ai"):
            # Forza la mappatura corretta per il giocatore 2 (cliente)
            correct_mapping = [2, 3, 0, 1]  # [0->2, 1->3, 2->0, 3->1]
            if self.visual_positions != correct_mapping:
                print("CORREZIONE CRITICA: Mappatura visuale forzata per il giocatore 2!")
                self.visual_positions = correct_mapping
        
        # Stampa dettagliata per debugging
        print(f"MAPPATURA VISUALE FINALE: {self.visual_positions}")
        print(f"Giocatore locale {self.local_player_id} posizionato in {self.visual_positions[self.local_player_id]}")
        for i in range(4):
            print(f"  Giocatore {i} mostrato in posizione {self.visual_positions[i]}")
        
        # BLOCCA la mappatura visuale per evitare modifiche successive
        setattr(self, '_visual_mapping_locked', True)

    def get_visual_position(self, logical_player_id):
        """Convert logical player ID to visual position based on client perspective"""
        if not hasattr(self, 'visual_positions') or self.local_player_id is None:
            return logical_player_id  # Use default mapping if no visual positions set yet
            
        # CRUCIALE: Verifica se qualcuno sta cercando di modificare il mapping
        if hasattr(self, '_visual_mapping_locked') and getattr(self, '_visual_mapping_locked'):
            # Stampa un avviso solo la prima volta
            if not hasattr(self, '_warned_about_mapping'):
                print(f"ATTENZIONE: Tentativo di accesso al mapping visuale bloccato")
                setattr(self, '_warned_about_mapping', True)
        
        # Stampa di debug ad ogni chiamata
        result = self.visual_positions[logical_player_id]
        #print(f"get_visual_position({logical_player_id}) => {result}")
        
        return result
    
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
        
        # Avatar positions between each hand and the table center, aligned to the hand
        avatar_size = int(height * 0.08)
        gap = int(width * 0.01)

        for player in self.players:
            visual_pos = self.get_visual_position(player.player_id)
            hand_rect = player.hand_rect

            if visual_pos == 0:  # Bottom player - above the hand toward table center
                player.avatar_rect = pygame.Rect(
                    hand_rect.centerx - avatar_size // 2,
                    hand_rect.top - avatar_size - gap,
                    avatar_size,
                    avatar_size,
                )
            elif visual_pos == 1:  # Left player - to the right of the hand toward table center
                player.avatar_rect = pygame.Rect(
                    hand_rect.right + gap,
                    hand_rect.centery - avatar_size // 2,
                    avatar_size,
                    avatar_size,
                )
            elif visual_pos == 2:  # Top player - below the hand toward table center
                player.avatar_rect = pygame.Rect(
                    hand_rect.centerx - avatar_size // 2,
                    hand_rect.bottom + gap,
                    avatar_size,
                    avatar_size,
                )
            elif visual_pos == 3:  # Right player - to the left of the hand toward table center
                player.avatar_rect = pygame.Rect(
                    hand_rect.left - avatar_size - gap,
                    hand_rect.centery - avatar_size // 2,
                    avatar_size,
                    avatar_size,
                )

            # Set info rect as the same as avatar rect for compatibility
            player.info_rect = player.avatar_rect.copy()
        
        # UI elements and other layout elements - similar to original
        button_width = int(width * 0.14)
        button_height = int(height * 0.06)
        
        # Exit button - top-left corner
        self.new_game_button = Button(
            width * 0.01,  # Far left
            height * 0.01,  # Far top
            button_width, 
            button_height,
            "Exit", 
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
        
        # Replay button - positioned above the Play Card button
        self.replay_button = Button(
            width - button_width - width * 0.01,
            height - button_height - height * 0.01 - button_height - height * 0.01,
            button_width, 
            button_height,
            "Replay Last 3", 
            DARK_BLUE, WHITE
        )
        
        # Message log placement
        msg_w = width * 0.25
        msg_h = height * 0.18
        margin = max(8, int(min(width, height) * 0.01))

        if self.app.game_config.get("mode") == "online_multiplayer":
            # Place at the corner between bottom of Player 2's hand and left of Player 3's hand
            try:
                player2 = next(p for p in self.players if p.player_id == 2)
                player3 = next(p for p in self.players if p.player_id == 3)
                corner_x = player3.hand_rect.left
                corner_y = player2.hand_rect.bottom

                # Position the box just inside the corner (left of player 3's hand, below player 2's hand)
                x = corner_x - msg_w - margin
                y = corner_y + margin

                # Clamp inside the window
                x = max(margin, min(x, width - msg_w - margin))
                y = max(margin, min(y, height - msg_h - margin))

                self.message_log_rect = pygame.Rect(x, y, msg_w, msg_h)
            except StopIteration:
                # Fallback to top-right if players not initialized
                self.message_log_rect = pygame.Rect(
                    width - msg_w - width * 0.02,
                    height * 0.05,
                    msg_w,
                    msg_h,
                )
        else:
            # Default: top right corner
            self.message_log_rect = pygame.Rect(
                width - msg_w - width * 0.02,
                height * 0.05,
                msg_w, 
                msg_h
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

    def get_team_pile_rect(self, team_id: int) -> pygame.Rect:
        """Return the rectangle where the team's captured cards pile should be drawn.

        - Local team (same team as local player): bottom-left corner
        - Opponent team: top-right corner; when AI difficulty is shown, place pile under it
        """
        width = self.app.window_width
        height = self.app.window_height

        pile_width = int(width * 0.14)
        pile_height = int(height * 0.14)
        margin_w = int(width * 0.02)
        margin_h = int(height * 0.02)

        local_team_id = self.players[self.local_player_id].team_id if hasattr(self, 'players') else 0
        is_local_team = (team_id == local_team_id)

        if is_local_team:
            # Bottom-left
            left = margin_w
            top = height - margin_h - pile_height
            return pygame.Rect(left, top, pile_width, pile_height)
        else:
            # Top-right; align under difficulty label when present
            top_base = int(height * 0.02)
            if self.app.game_config.get("mode") in ["single_player", "team_vs_ai"]:
                top_base = int(height * 0.026) + self.small_font.get_height() + int(height * 0.012)
            left = width - margin_w - pile_width
            top = top_base
            return pygame.Rect(left, top, pile_width, pile_height)
    
    def update_player_hands(self):
        """Update player hand information from game state in a safe way"""
        if not self.env:
            return
        
        gs = getattr(self.env, 'game_state', None)
        if not isinstance(gs, dict):
            # Clear all hands if no valid game state is present yet
            for player in self.players:
                player.set_hand([])
            return
        
        hands = gs.get('hands', {})
        if not isinstance(hands, dict):
            hands = {}
        
        for player in self.players:
            hand = hands.get(player.player_id, [])
            player.set_hand(hand if isinstance(hand, list) else [])
    def handle_events(self, events):
        """Handle pygame events with connection loss recovery"""
        for event in events:
            if event.type == pygame.QUIT:
                self.done = True
                self.next_screen = None
                pygame.quit()
                sys.exit()
            
            # Check exit button when connection is lost
            if self.connection_lost and self.exit_button:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if self.exit_button.is_clicked(pos):
                        self.done = True
                        self.next_screen = "mode"
                        return
            
            # Schermata di recap intermedio (modalità a punti/mani)
            if self.show_intermediate_recap and self.next_hand_button:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    if self.next_hand_button.is_clicked(pos):
                        # Avanza alla mano successiva
                        if self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host"):
                            next_starter = self._pending_next_starter if self._pending_next_starter is not None else random.randint(0, 3)
                            self.series_prev_starter = next_starter
                            self.game_over = False
                            self.final_breakdown = None
                            self.status_message = ""
                            self.selected_hand_card = None
                            self.selected_table_cards = set()
                            self.animations = []
                            rules = self.app.game_config.get("rules", {})
                            self.env = ScoponeEnvMA(rules=rules)
                            self.env.reset(starting_player=next_starter)
                            self.setup_players()
                            self.setup_layout()
                            # Chiudi overlay
                            self.show_intermediate_recap = False
                            self.last_hand_breakdown = None
                            self._pending_next_starter = None
                            self.next_hand_button = None
                            self.exit_overlay_button = None
                            # Broadcast nuovo stato per i client
                            self._broadcast_series_state()
                            return
                        else:
                            # Client: aspetta host; non deve avanzare
                            return
                    if getattr(self, 'exit_overlay_button', None) and self.exit_overlay_button.is_clicked(pos):
                        # Esci al menu
                        self.done = True
                        self.next_screen = "mode"
                        return

            # Always check for the Exit button and message log interactions
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                
                if self.new_game_button.is_clicked(pos):
                    self.done = True
                    self.next_screen = "mode"
                    return

                # Message log interactions (active regardless of game state)
                if self.message_log_rect.collidepoint(pos):
                    self.message_focused = True
                    # Minimize toggle
                    if self.message_minimize_rect and self.message_minimize_rect.collidepoint(pos):
                        # Toggle minimize and adjust size
                        if not self.message_minimized:
                            # Save previous height and minimize to header height
                            self.message_prev_height = self.message_log_rect.height
                            self.message_log_rect.height = self.message_header_height + 2  # show border
                            self.message_minimized = True
                        else:
                            # Restore height
                            if self.message_prev_height:
                                self.message_log_rect.height = self.message_prev_height
                            self.message_minimized = False
                        return
                    # Start dragging if click in header
                    header_rect = pygame.Rect(
                        self.message_log_rect.left,
                        self.message_log_rect.top,
                        self.message_log_rect.width,
                        self.message_header_height,
                    )
                    if header_rect.collidepoint(pos):
                        self.message_dragging = True
                        mx, my = pos
                        self.message_drag_offset = (mx - self.message_log_rect.left, my - self.message_log_rect.top)
                        return
                    # Scrollbar clicks
                    if self.scroll_up_rect and self.scroll_up_rect.collidepoint(pos):
                        self.message_scroll_offset = max(0, self.message_scroll_offset - 1)
                        return
                    if self.scroll_down_rect and self.scroll_down_rect.collidepoint(pos):
                        self.message_scroll_offset = min(self.message_scroll_offset + 1, getattr(self, 'scrollbar_max_offset', self.message_scroll_offset + 1))
                        return
                    if self.scrollbar_thumb_rect and self.scrollbar_thumb_rect.collidepoint(pos):
                        self.scrollbar_dragging = True
                        self.scrollbar_drag_offset = pos[1] - self.scrollbar_thumb_rect.top
                        return
                    if self.scrollbar_rect and self.scrollbar_rect.collidepoint(pos):
                        # Jump to position and start dragging
                        my = pos[1]
                        track_top = self.scrollbar_rect.top
                        track_height = self.scrollbar_rect.height
                        rel_y = my - track_top - self.scrollbar_thumb_height / 2
                        rel_y = max(0, min(rel_y, track_height - self.scrollbar_thumb_height))
                        if self.scrollbar_max_offset > 0:
                            ratio = rel_y / (track_height - self.scrollbar_thumb_height)
                            self.message_scroll_offset = int(round(ratio * self.scrollbar_max_offset))
                        self.scrollbar_dragging = True
                        self.scrollbar_drag_offset = self.scrollbar_thumb_height / 2
                        return
                # Start resizing if click in resize handle (only when not minimized)
                if (not self.message_minimized) and self.message_resize_rect and self.message_resize_rect.collidepoint(pos):
                    self.message_resizing = True
                    self.message_resize_start_mouse = pos
                    self.message_resize_start_rect = self.message_log_rect.copy()
                    return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Click outside message box removes focus
                pos = pygame.mouse.get_pos()
                if not self.message_log_rect.collidepoint(pos):
                    self.message_focused = False
            
            # Ignore input during specific game states
            if self.game_over or self.waiting_for_other_player or self.ai_thinking:
                if self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if new game button is clicked in game over screen
                    mouse_pos = pygame.mouse.get_pos()
                    if hasattr(self, 'game_over_button_rect') and self.game_over_button_rect and self.game_over_button_rect.collidepoint(mouse_pos):
                        self.done = True
                        self.next_screen = "mode"
                # Allow message log drag/resize while other inputs are ignored
                if event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if self.message_dragging:
                        new_x = mx - self.message_drag_offset[0]
                        new_y = my - self.message_drag_offset[1]
                        new_x = max(0, min(new_x, self.app.window_width - self.message_log_rect.width))
                        new_y = max(0, min(new_y, self.app.window_height - self.message_log_rect.height))
                        self.message_log_rect.topleft = (new_x, new_y)
                        return
                    if self.message_resizing:
                        start_x, start_y = self.message_resize_start_mouse
                        dx = mx - start_x
                        dy = my - start_y
                        min_w = int(self.app.window_width * 0.15)
                        min_h = int(self.app.window_height * 0.12)
                        new_w = max(min_w, self.message_resize_start_rect.width + dx)
                        new_h = max(min_h, self.message_resize_start_rect.height + dy)
                        max_w = int(self.app.window_width * 0.6)
                        max_h = int(self.app.window_height * 0.6)
                        new_w = min(new_w, max_w)
                        new_h = min(new_h, max_h)
                        self.message_log_rect.size = (new_w, new_h)
                        return
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.message_dragging = False
                        self.message_resizing = False
                continue
            
            # Check if current player is controllable
            if not self.is_current_player_controllable():
                continue
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                # Process confirm button
                if self.confirm_button.is_clicked(pos):
                    self.try_make_move()
                
        # Process mouse down actions (replay button and selections)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = pygame.mouse.get_pos()
            # Replay button click only on actual click (not hover)
            if self.replay_button.is_clicked(pos):
                if not self.animations and not self.replay_active:
                    self.start_replay()
                # Do not fall-through to card selection when clicking the replay button
                return
            
            # Card selection in hand
            hand_card = self.get_card_at_position(pos, area="hand")
            if hand_card:
                if hand_card == self.selected_hand_card:
                    self.selected_hand_card = None
                else:
                    self.selected_hand_card = hand_card
                self.app.resources.play_sound("card_pickup")
                return
            
            # Card selection on table
            table_card = self.get_card_at_position(pos, area="table")
            if table_card:
                if table_card in self.selected_table_cards:
                    self.selected_table_cards.remove(table_card)
                else:
                    self.selected_table_cards.add(table_card)
                self.app.resources.play_sound("card_pickup")
                return
            
            # Handle message log scrolling (legacy wheel buttons 4/5)
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                # Check scroll button clicks (only exist when scrollbar is visible) handled earlier in message box section.
                
                # Mouse wheel over the message log (classic wheel buttons 4/5)
                if self.message_log_rect.collidepoint(pos) or self.message_focused:
                    if event.button == 4:  # Scroll up
                        self.message_scroll_offset = max(0, self.message_scroll_offset - 1)
                    elif event.button == 5:  # Scroll down
                        self.message_scroll_offset += 1

            # Keyboard scrolling when message box focused
            if event.type == pygame.KEYDOWN and self.message_focused:
                if event.key in (pygame.K_UP, pygame.K_PAGEUP):
                    self.message_scroll_offset = max(0, self.message_scroll_offset - 1)
                elif event.key in (pygame.K_DOWN, pygame.K_PAGEDOWN):
                    self.message_scroll_offset += 1

            # Support modern mouse wheel events (pygame 2)
            if hasattr(pygame, 'MOUSEWHEEL') and event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                # Allow wheel when hovering or when focused
                if self.message_log_rect.collidepoint(mouse_pos) or self.message_focused:
                    # event.y: 1 = up, -1 = down
                    self.message_scroll_offset = max(0, self.message_scroll_offset - event.y)

            # Dragging / Resizing while normal play
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self.message_dragging:
                    new_x = mx - self.message_drag_offset[0]
                    new_y = my - self.message_drag_offset[1]
                    new_x = max(0, min(new_x, self.app.window_width - self.message_log_rect.width))
                    new_y = max(0, min(new_y, self.app.window_height - self.message_log_rect.height))
                    self.message_log_rect.topleft = (new_x, new_y)
                if self.message_resizing:
                    start_x, start_y = self.message_resize_start_mouse
                    dx = mx - start_x
                    dy = my - start_y
                    min_w = int(self.app.window_width * 0.15)
                    min_h = int(self.app.window_height * 0.12)
                    new_w = max(min_w, self.message_resize_start_rect.width + dx)
                    new_h = max(min_h, self.message_resize_start_rect.height + dy)
                    max_w = int(self.app.window_width * 0.6)
                    max_h = int(self.app.window_height * 0.6)
                    new_w = min(new_w, max_w)
                    new_h = min(new_h, max_h)
                    self.message_log_rect.size = (new_w, new_h)
                # Scrollbar dragging
                if self.scrollbar_dragging and self.scrollbar_rect:
                    # map mouse y to scroll offset
                    track_top = self.scrollbar_rect.top
                    track_height = self.scrollbar_rect.height
                    rel_y = my - track_top - self.scrollbar_drag_offset
                    rel_y = max(0, min(rel_y, track_height - self.scrollbar_thumb_height))
                    if self.scrollbar_max_offset > 0:
                        ratio = rel_y / (track_height - self.scrollbar_thumb_height)
                        self.message_scroll_offset = int(round(ratio * self.scrollbar_max_offset))
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.message_dragging = False
                    self.message_resizing = False
                    self.scrollbar_dragging = False

            # Dragging / Resizing while normal play
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self.message_dragging:
                    new_x = mx - self.message_drag_offset[0]
                    new_y = my - self.message_drag_offset[1]
                    new_x = max(0, min(new_x, self.app.window_width - self.message_log_rect.width))
                    new_y = max(0, min(new_y, self.app.window_height - self.message_log_rect.height))
                    self.message_log_rect.topleft = (new_x, new_y)
                if self.message_resizing:
                    start_x, start_y = self.message_resize_start_mouse
                    dx = mx - start_x
                    dy = my - start_y
                    min_w = int(self.app.window_width * 0.15)
                    min_h = int(self.app.window_height * 0.12)
                    new_w = max(min_w, self.message_resize_start_rect.width + dx)
                    new_h = max(min_h, self.message_resize_start_rect.height + dy)
                    max_w = int(self.app.window_width * 0.6)
                    max_h = int(self.app.window_height * 0.6)
                    new_w = min(new_w, max_w)
                    new_h = min(new_h, max_h)
                    self.message_log_rect.size = (new_w, new_h)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.message_dragging = False
                    self.message_resizing = False
    
    def update(self):
        """Update game state with improved animation handling"""
        # Update player hands from game state
        self.update_player_hands()
        
        # Update current player
        if self.env:
            self.current_player_id = self.env.current_player
            # Reset one-shot auto-play guard when turn changes
            if getattr(self, '_last_player_for_autoplay', None) != self.current_player_id:
                self.autoplay_sent_for_turn = False
                self._last_player_for_autoplay = self.current_player_id
        
        # Debug: stampa numero di animazioni attive
        if hasattr(self, 'animations') and self.animations:
            #print(f"Animazioni attive: {len(self.animations)}")
            for idx, anim in enumerate(self.animations[:5]):  # Mostra solo le prime 5 per brevità
                status = "ritardo" if anim.current_frame < 0 else "attiva" if not anim.done else "completata"
                #print(f"  Anim {idx}: carta {anim.card}, tipo {anim.animation_type}, stato {status}, frame {anim.current_frame}/{anim.duration}")
        
        # Update animations
        active_animations = self.update_animations()
        
        # Update replay animations if replay is active (single source of truth for table state during replay)
        if self.replay_active:
            self.update_replay_animations()
        
        # FIXED: Handling of animation phases
        if hasattr(self, 'waiting_for_animation') and self.waiting_for_animation:
            # Check if all animations are completed
            if not self.animations:  # No active animations means all are completed
                #print("Tutte le animazioni completate, ora aggiorno lo stato del gioco")
                
                # Check for pending action and execute it
                if hasattr(self, 'pending_action') and self.pending_action is not None:
                    # CRITICAL FIX: Verify the action is valid for the current player before executing
                    card_played, _ = decode_action(self.pending_action)
                    hands = {}
                    if hasattr(self.env, 'game_state') and isinstance(self.env.game_state, dict):
                        hands = self.env.game_state.get('hands', {}) if isinstance(self.env.game_state.get('hands', {}), dict) else {}
                    current_player_hand = hands.get(self.env.current_player, [])
                    
                    if card_played in current_player_hand:
                        # Execute the move on the environment
                        _, _, done, info = self.env.step(self.pending_action)
                        
                        # If game is finished, set final state
                        if done:
                            self.game_over = True
                            if "score_breakdown" in info:
                                self.final_breakdown = info["score_breakdown"]
                    else:
                        # The card is no longer valid - this can happen if turn changed while animation was playing
                        print(f"WARNING: Card {card_played} not in player {self.env.current_player}'s hand - skipping action")
                    
                    # IMPORTANT: After updating state, reset hidden cards visually
                    if hasattr(self, 'visually_hidden_cards'):
                        self.visually_hidden_cards = {}
                    
                    # Reset animation states
                    self.waiting_for_animation = False
                    self.pending_action = None
                    
                    # Update game state for online mode (if needed)
                    if self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host"):
                        if hasattr(self.app, 'network') and self.app.network:
                            self._broadcast_series_state()
                    
                    # Detect end of hand after environment update
                    if hasattr(self.env, 'done') and self.env.done:
                        if self.app.game_config.get("mode") == "online_multiplayer":
                            # Host gestisce la serie e la sincronizza
                            if self.app.game_config.get("is_host"):
                                self._handle_hand_end(info.get("score_breakdown"))
                                # Broadcast stato serie/recap
                                if hasattr(self, 'app') and hasattr(self.app, 'network') and self.app.network:
                                    self._broadcast_series_state()
                        else:
                            self._handle_hand_end(info.get("score_breakdown"))
        
        # Check for connection loss in online mode
        if (self.app.game_config.get("mode") == "online_multiplayer" and 
                self.app.network and not self.app.network.connected):
            
            # If connection was just lost
            if not self.connection_lost:
                self.connection_lost = True
                self.reconnect_start_time = time.time()
                self.create_exit_button()
                self.status_message = "Connection to host lost. Attempting to reconnect..."
                self.reconnection_attempts = 0
                self.app.resources.play_sound("card_pickup")  # Alert sound
            
            # Check if we should attempt reconnection
            current_time = time.time()
            elapsed = current_time - self.reconnect_start_time
            
            # Try to reconnect every 5 seconds
            if elapsed > self.reconnection_attempts * 5 and not self.app.network.connection_in_progress:
                self.reconnection_attempts += 1
                # Start async reconnection attempt
                self.app.network.connection_in_progress = True
                self.app.network.connection_start_time = time.time()
                threading.Thread(target=self.app.network._connect_async, daemon=True).start()
            
            # If reconnection successful
            if self.app.network.connected:
                self.connection_lost = False
                self.status_message = "Reconnected to host!"
                self.exit_button = None
                # Request fresh game state
                if self.app.network.is_host:
                    self.app.network.broadcast_game_state()
            
            # If timeout reached (1 minute)
            elif elapsed > self.max_reconnection_time:
                self.status_message = "Connection timeout. Returning to main menu..."
                # Exit after a short delay
                pygame.time.delay(2000)
                self.done = True
                self.next_screen = "mode"
        
        # Continue with regular updates if connection is fine
        if not self.connection_lost:
        
            # Aggiorna local_player_id dal network se necessario
            if self.app.game_config.get("mode") == "online_multiplayer" and not self.app.game_config.get("is_host"):
                if self.app.network and self.app.network.player_id is not None:
                    old_player_id = self.local_player_id
                    if self.local_player_id != self.app.network.player_id:
                        print(f"Aggiornamento player_id: da {self.local_player_id} a {self.app.network.player_id}")
                        self.local_player_id = self.app.network.player_id
                        
                        # Aggiorna i giocatori in caso di cambio ID
                        if old_player_id != self.local_player_id:
                            # Riaggiorna player info, prospettiva e layout
                            self.setup_players()
                            self.setup_player_perspective()
                            self.setup_layout()
            
            # Auto-play when a player has exactly one card left, unless multiple capture choices exist (humans only)
            try:
                if (self.env and not self.game_over and not self.animations
                        and not getattr(self, 'waiting_for_animation', False)
                        and not getattr(self, 'show_intermediate_recap', False)
                        and not getattr(self, 'autoplay_sent_for_turn', False)):
                    mode = self.app.game_config.get("mode")
                    is_online = (mode == "online_multiplayer")
                    is_host = self.app.game_config.get("is_host", False)
                    cp = self.env.current_player
                    hands = {}
                    if hasattr(self.env, 'game_state') and isinstance(self.env.game_state, dict):
                        hands = self.env.game_state.get('hands', {}) if isinstance(self.env.game_state.get('hands', {}), dict) else {}
                    hand = hands.get(cp, [])
                    if len(hand) == 1:
                        the_card = hand[0]
                        valid_actions = self.env.get_valid_actions() or []
                        # Filter actions for the only card in hand
                        filtered = []
                        for act in valid_actions:
                            try:
                                pc, cc = decode_action(act)
                                if pc == the_card:
                                    filtered.append((act, pc, cc))
                            except Exception:
                                continue
                        if filtered:
                            capture_options = [t for t in filtered if t[2]]
                            multiple_captures = len(capture_options) > 1
                            current_is_human = not self.players[self.current_player_id].is_ai
                            # Choose action we would auto-play with
                            if capture_options:
                                chosen = capture_options[0]
                            else:
                                no_cap = next((t for t in filtered if not t[2]), None)
                                if no_cap is None:
                                    no_cap = filtered[0]
                                chosen = no_cap
                            action, card_played, cards_captured = chosen
                            if is_online and not is_host:
                                # Client: auto-send move only if human and not multiple capture choices
                                if current_is_human and not multiple_captures and cp == self.local_player_id:
                                    if hasattr(self.app, 'network') and self.app.network:
                                        self.app.network.send_move(action)
                                        self.autoplay_sent_for_turn = True
                            else:
                                # Local or Host: perform animations and schedule env step
                                if (not multiple_captures) or (not current_is_human):
                                    self.create_move_animations(card_played, cards_captured)
                                    self.app.resources.play_sound("card_play")
                                    self.waiting_for_animation = True
                                    self.pending_action = action
                                    self.autoplay_sent_for_turn = True
            except Exception as e:
                # Non bloccare il gioco in caso di errore nell'auto-play
                print(f"Auto-play error: {e}")

            # Handle AI turns (solo il server lo fa in modalità online)
            self.handle_ai_turns()
            
            # Handle network updates
            self.handle_network_updates()
            
            # Check for game over
            if self.env and not self.game_over:
                self.check_game_over()
            
            # CORREZIONE: Forza la configurazione del team vs AI all'inizio del gioco
            is_online_team_vs_ai = (self.app.game_config.get("mode") == "online_multiplayer" and 
                                    self.app.game_config.get("online_type") == "team_vs_ai")
            
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
                        
                        # NUOVO: Trasmette immediatamente lo stato al client per sincronizzare le informazioni
                        self.perform_initial_sync()
                
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
                elif online_type == "humans_plus_ai":
                    # Attendi 1 umano (client). Ogni umano avrà un'AI compagna.
                    needed_players = 1
                    if self.app.network and len(self.app.network.clients) < needed_players:
                        connected_count = len(self.app.network.clients)
                        self.status_message = f"Server attivo su {ip_address}:5555 | {connected_count}/{needed_players} giocatori connessi"
                        self.waiting_for_other_player = True
                    elif self.app.network and len(self.app.network.clients) == needed_players and self.waiting_for_other_player:
                        # Configura squadre: 0 (umano host) + AI 2 contro 1 (umano client) + AI 3
                        self.setup_humans_plus_ai_online()
                        self.waiting_for_other_player = False
                        self.status_message = "Avversario connesso. Inizia il gioco!"
                elif online_type == "three_humans_one_ai":
                    # Attendi 2 umani (client). I 3 posti umani si scelgono in lobby; l'altro posto sarà AI.
                    needed_players = 2
                    if self.app.network and len(self.app.network.clients) < needed_players:
                        connected_count = len(self.app.network.clients)
                        self.status_message = f"Server attivo su {ip_address}:5555 | {connected_count}/{needed_players} giocatori connessi"
                        self.waiting_for_other_player = True
                    elif self.app.network and len(self.app.network.clients) == needed_players and self.waiting_for_other_player:
                        # La configurazione AI specifica avverrà in setup al primo sync partita
                        self.waiting_for_other_player = False
                        self.status_message = "Tutti gli umani connessi. Inizia il gioco!"
            
            # CORREZIONE: Assicura che anche il client configuri correttamente i giocatori AI
            elif is_online_team_vs_ai and not self.app.game_config.get("is_host"):
                # Controllo se i giocatori AI sono già configurati
                ai_players_configured = any(player.is_ai for player in self.players)
                
                # Se non ancora configurati, configura i giocatori AI
                if not ai_players_configured:
                    print("Client: configurazione giocatori AI")
                    # Gli stessi passaggi di setup_team_vs_ai_online() ma per il client
                    # Configura giocatori umani
                    self.players[0].team_id = 0
                    self.players[0].is_human = True
                    self.players[0].is_ai = False
                    self.players[0].name = "Partner" if self.local_player_id == 2 else "You"
                    
                    self.players[2].team_id = 0
                    self.players[2].is_human = True
                    self.players[2].is_ai = False
                    self.players[2].name = "You" if self.local_player_id == 2 else "Partner"
                    
                    # Configura giocatori AI
                    self.players[1].team_id = 1
                    self.players[1].is_human = False
                    self.players[1].is_ai = True
                    self.players[1].name = "AI 1"
                    
                    self.players[3].team_id = 1
                    self.players[3].is_human = False
                    self.players[3].is_ai = True
                    self.players[3].name = "AI 3"
            
            # Process messages from network
            if hasattr(self, 'app') and hasattr(self.app, 'network') and self.app.network:
                while self.app.network.message_queue:
                    message = self.app.network.message_queue.popleft()
                    self.messages.append(message)
                    
    def perform_initial_sync(self):
        """Invia lo stato iniziale completo dal server al client, inclusi mazzo e giocatore iniziale"""
        if not hasattr(self.app, 'network') or not self.app.network:
            return
        
        if not self.app.game_config.get("is_host", False):
            return  # Solo l'host esegue la sincronizzazione iniziale
        
        print("SYNC: Invio stato iniziale completo al client")
        
        # Assicurati che lo stato includa tutte le informazioni necessarie
        self.app.network.game_state = self.env.game_state.copy()
        
        # Aggiungi informazioni aggiuntive al game_state
        online_type = self.app.game_config.get("online_type")
        if online_type:
            self.app.network.game_state['online_type'] = online_type
            if online_type == "team_vs_ai":
                self.app.network.game_state['ai_players'] = [1, 3]
        
        # Includi esplicitamente informazioni sul giocatore corrente
        self.app.network.game_state['current_player'] = self.env.current_player
        
        # Includi il mazzo di carte completo (con ordine casuale determinato dall'host)
        if hasattr(self.env, 'deck'):
            self.app.network.game_state['deck'] = self.env.deck
        
        # Trasmetti lo stato completo
        self.app.network.broadcast_game_state()
        print(f"SYNC: Stato iniziale inviato - giocatore corrente: {self.env.current_player}")


    def setup_team_vs_ai_online(self):
        """Configurazione robusta dei giocatori per la modalità Team vs AI online"""
        print("\n### CONFIGURAZIONE TEAM VS AI ONLINE ###")
        print(f"ID giocatore locale: {self.local_player_id}")
        
        # Configurazione fissa per squadre e ruoli
        # Team 0 (umani): giocatori 0 e 2
        # Team 1 (AI): giocatori 1 e 3
        
        # FASE 1-3: Configura squadre, ruoli e nomi (invariato)
        self.players[0].team_id = 0  # Team umano
        self.players[2].team_id = 0  # Team umano
        self.players[1].team_id = 1  # Team AI
        self.players[3].team_id = 1  # Team AI
        
        self.players[0].is_human = True
        self.players[0].is_ai = False
        
        self.players[2].is_human = True
        self.players[2].is_ai = False
        
        self.players[1].is_human = False
        self.players[1].is_ai = True
        
        self.players[3].is_human = False
        self.players[3].is_ai = True
        
        if self.local_player_id == 0:
            self.players[0].name = "You"
            self.players[2].name = "Partner"
        else:  # self.local_player_id == 2
            self.players[0].name = "Partner"
            self.players[2].name = "You"
        
        self.players[1].name = "AI 1"
        self.players[3].name = "AI 3"
        
        # FASE 4: Configura UN SOLO AI controller per la squadra 1
        # Controlla se l'agente è già stato creato
        ai_controller_exists = 1 in self.ai_controllers and 3 in self.ai_controllers
        same_agent = ai_controller_exists and (self.ai_controllers[1] is self.ai_controllers[3])
        
        if not (ai_controller_exists and same_agent):
            print(f"Creazione controller AI unificato per la squadra 1")
            ai_agent = DQNAgent(team_id=1)
            
            # Carica checkpoint se disponibile
            checkpoint_path = f"scopone_checkpoint_team1.pth"
            if os.path.exists(checkpoint_path):
                print(f"Caricamento checkpoint per la squadra AI")
                ai_agent.load_checkpoint(checkpoint_path)
            
            # Imposta epsilon in base alla difficoltà
            difficulty = self.ai_difficulty
            if difficulty == 0:  # Easy
                ai_agent.epsilon = 0.3
            elif difficulty == 1:  # Medium
                ai_agent.epsilon = 0.1
            else:  # Hard
                ai_agent.epsilon = 0
                
            # Assegna lo stesso agente a entrambi i giocatori AI
            self.ai_controllers[1] = ai_agent
            self.ai_controllers[3] = ai_agent
        else:
            print(f"Controller AI unificato per la squadra 1 già esistente")
        
        # FASE 5: Imposta in modo esplicito i giocatori AI nel game_state
        if hasattr(self.app, 'network') and self.app.network:
            self.perform_initial_sync()
        
        # Resto del codice invariato...
        print("TEAM UMANO: giocatori 0 e 2")
        print("TEAM AI: giocatori 1 e 3")
        self.messages.append("TEAM UMANO: giocatori 0 e 2") 
        self.messages.append("TEAM AI: giocatori 1 e 3")
        
        print("\nConfigurazione finale dei giocatori:")
        for player in self.players:
            print(f"Player {player.player_id}: {player.name}, Team {player.team_id}, AI: {player.is_ai}")
            
        self.waiting_for_other_player = False

    def setup_humans_plus_ai_online(self):
        """Configurazione per online_type=humans_plus_ai: due umani su squadre opposte con AI compagna."""
        print("\n### CONFIGURAZIONE HUMANS + AI ONLINE ###")
        print(f"ID giocatore locale: {self.local_player_id}")

        # Squadre: (0 umano host + AI 2) vs (1 umano client + AI 3)
        self.players[0].team_id = 0
        self.players[2].team_id = 0
        self.players[1].team_id = 1
        self.players[3].team_id = 1

        # Ruoli umani/AI
        self.players[0].is_human = True
        self.players[0].is_ai = False
        self.players[1].is_human = True
        self.players[1].is_ai = False
        self.players[2].is_human = False
        self.players[2].is_ai = True
        self.players[3].is_human = False
        self.players[3].is_ai = True

        # Nomi dipendenti dal punto di vista locale
        if self.local_player_id == 0:
            self.players[0].name = "You"
            self.players[1].name = "Opponent"
        elif self.local_player_id == 1:
            self.players[1].name = "You"
            self.players[0].name = "Opponent"
        else:
            # Default
            self.players[0].name = "Host"
            self.players[1].name = "Client"
        self.players[2].name = "AI 2"
        self.players[3].name = "AI 3"

        # Controller AI: un agente per team 0? No, solo per i giocatori AI (2 e 3) con stesso livello
        # Creiamo un agente per ogni team AI in base alla difficoltà, ma sono indipendenti
        difficulty = self.ai_difficulty
        ai_agent_team0 = DQNAgent(team_id=0)
        ai_agent_team1 = DQNAgent(team_id=1)
        # Caricamento checkpoint opzionale
        ck0 = "scopone_checkpoint_team0.pth"
        ck1 = "scopone_checkpoint_team1.pth"
        if os.path.exists(ck0):
            ai_agent_team0.load_checkpoint(ck0)
        if os.path.exists(ck1):
            ai_agent_team1.load_checkpoint(ck1)
        # Stessa epsilon in base al livello per entrambi
        if difficulty == 0:
            ai_agent_team0.epsilon = ai_agent_team1.epsilon = 0.3
        elif difficulty == 1:
            ai_agent_team0.epsilon = ai_agent_team1.epsilon = 0.1
        else:
            ai_agent_team0.epsilon = ai_agent_team1.epsilon = 0

        # Assegna controller ai giocatori AI
        self.ai_controllers[2] = ai_agent_team0  # AI compagno host
        self.ai_controllers[3] = ai_agent_team1  # AI compagno client

        # Sync tipo online e stato iniziale
        if hasattr(self.app, 'network') and self.app.network and self.app.game_config.get("is_host"):
            self.perform_initial_sync()
            self.app.network.game_state['online_type'] = 'humans_plus_ai'
            # Comunica quali giocatori sono AI
            self.app.network.game_state['ai_players'] = [2, 3]

        print("Squadre: (0,2) vs (1,3) con AI in 2 e 3")
        self.messages.append("Team 0: Human 0 + AI 2")
        self.messages.append("Team 1: Human 1 + AI 3")
    
    def update_animations(self):
        """Update and clean up animations, return True if animations are active"""
        active_animations = False
        
        # NUOVO: Assicurati che esista il set di carte in movimento
        if not hasattr(self, 'cards_in_motion'):
            self.cards_in_motion = set()
        
        # NUOVO: Lista temporanea per le carte da rimuovere dal set di movimento
        cards_to_remove = []
        
        # Update existing animations (skip during replay to avoid duplicate animations)
        if self.replay_active:
            return False
        for anim in self.animations[:]:
            # Verifica se questa è un'animazione di inizio movimento
            if anim.animation_type == "start_motion" and anim.current_frame == 0:
                # Aggiungi la carta al set delle carte in movimento
                self.cards_in_motion.add(anim.card)
                #print(f"Carta {anim.card} aggiunta al set di carte in movimento")
            
            # Verifica se questa è un'animazione di cattura che termina
            if anim.animation_type == "capture" and anim.current_frame == anim.duration - 1:
                # Aggiungi la carta alla lista di rimozione invece di rimuoverla immediatamente
                cards_to_remove.append(anim.card)
                #print(f"Carta {anim.card} marcata per rimozione dal set di carte in movimento")
            
            # Debug dettagliato
            #if hasattr(anim, 'card') and hasattr(anim, 'animation_type') and hasattr(anim, 'current_frame'):
                # Debug solo per frame specifici
                #if anim.current_frame == 0 or anim.current_frame == anim.duration - 1:
                    #print(f"Animazione {anim.animation_type} per carta {anim.card}: frame {anim.current_frame}/{anim.duration}")
            
            # Aggiorna animazione
            is_completed = anim.update()
            
            # Rimuovi solo se completata
            if is_completed:
                #if hasattr(anim, 'card'):
                    #print(f"Animazione completata per carta {anim.card}, tipo {anim.animation_type}")
                self.animations.remove(anim)
            else:
                active_animations = True
        
        # NUOVO: Rimuovi le carte dal set di movimento quando non esiste più
        # alcuna animazione per quella carta, indipendentemente dal fatto che
        # ci siano altre animazioni attive per altre carte
        if cards_to_remove:
            for card in cards_to_remove:
                # Verifica che non esistano altre animazioni attive per questa carta
                still_animating = any(getattr(anim, 'card', None) == card and not getattr(anim, 'done', False)
                                      for anim in self.animations)
                if not still_animating and card in self.cards_in_motion:
                    self.cards_in_motion.remove(card)
                    #print(f"Carta {card} rimossa dal set di carte in movimento (dopo completamento animazioni)")
        
        return active_animations
    
    def handle_ai_turns(self):
        """Handle turns for AI-controlled players"""
        if not self.env or self.game_over or self.animations:
            return
            
        # Check if it's AI's turn
        current_player = self.players[self.current_player_id]
        
        # In modalità online, solo l'host gestisce le mosse delle AI
        is_online = self.app.game_config.get("mode") == "online_multiplayer"
        if is_online and not self.app.game_config.get("is_host", False):
            # I client non fanno nulla, riceveranno aggiornamenti dall'host
            return
        
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
            base_delay = 2000 if self.ai_difficulty == 0 else 1000 if self.ai_difficulty == 1 else 500
            delay = base_delay
            if current_time - self.ai_move_timer > delay:
                self.make_ai_move()
                self.ai_thinking = False
    def handle_network_updates(self):
        """Network update handling with improved state synchronization"""
        if not self.app.network:
            return
        
        # Host logic: process moves and broadcast state
        if self.app.network.is_host:
            # SOLUZIONE: Assicuriamo che il client sappia sempre chi sia il giocatore corrente
            if hasattr(self, 'initial_sync_done') and not self.initial_sync_done:
                print("SYNC: Invio sincronizzazione iniziale al client")
                self.app.network.game_state = self.env.game_state.copy()
                self.app.network.game_state['current_player'] = self.env.current_player
                
                # Se siamo in modalità team_vs_ai, aggiungi informazioni specifiche
                if self.app.game_config.get("online_type") == "team_vs_ai":
                    self.app.network.game_state['online_type'] = 'team_vs_ai'
                    self.app.network.game_state['ai_players'] = [1, 3]
                
                # Broadcast immediato
                self.app.network.broadcast_game_state()
                self.initial_sync_done = True
            
            # Process any queued moves
            while self.app.network.move_queue:
                player_id, move = self.app.network.move_queue.popleft()
                
                # Verify it's the player's turn
                if player_id != self.env.current_player:
                    print(f"Ignoring move from player {player_id}, it's player {self.env.current_player}'s turn")
                    continue
                
                # Execute the move and create animations
                try:
                    # Decode the move
                    card_played, cards_captured = decode_action(move)
                    #print(f"Host processing move: Player {player_id} plays {card_played}, captures {cards_captured}")
                    
                    # Create animations for the move
                    self.create_move_animations(card_played, cards_captured, player_id)
                    
                    # Execute the move in the environment
                    _, _, done, info = self.env.step(move)
                    
                    # Update game/series if hand is over (host)
                    if done:
                        self.game_over = True
                        if "score_breakdown" in info:
                            self.final_breakdown = info["score_breakdown"]
                            # Host: advance series logic and broadcast overlay/state
                            try:
                                self._handle_hand_end(info.get("score_breakdown"))
                            except Exception:
                                pass
                    
                    # ENHANCED: Prepare a deep copy of the game state for broadcasting
                    online_type = self.app.game_config.get("online_type")
                    self.app.network.game_state = self.env.game_state.copy()
                    if online_type:
                        self.app.network.game_state['online_type'] = online_type
                        self.app.network.game_state['ai_players'] = [1, 3]  # In team vs AI mode, players 1 and 3 are always AI
                    
                    # Add the current player to the state
                    self.app.network.game_state['current_player'] = self.env.current_player
                    
                    # Add information about the move that was just made
                    self.app.network.game_state['last_move'] = {
                        'player': player_id,
                        'card_played': card_played,
                        'cards_captured': cards_captured
                    }
                    
                    # Use the enhanced broadcast method
                    self.app.network.broadcast_game_state()
                    
                    # Play a sound effect
                    self.app.resources.play_sound("card_play")
                    
                    # NEW: Diagnostic message
                    table_cards = self.env.game_state.get("table", [])
                    print(f"Host broadcasting after player move - Current player: {self.env.current_player}, Table: {table_cards}")

                    # Also broadcast current series state for clients (overlay, scores, etc.)
                    try:
                        self._broadcast_series_state()
                    except Exception:
                        pass
                    
                except Exception as e:
                    print(f"Error processing move: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Client logic: update local state from network
        else:
            if self.app.network.game_state:
                # Keep a copy of old state for comparison
                old_state = None
                old_current_player = None
                if self.env and hasattr(self.env, 'game_state'):
                    old_state = {k: v.copy() if isinstance(v, dict) or isinstance(v, list) else v 
                                for k, v in self.env.game_state.items()} if self.env.game_state else None
                    old_current_player = self.env.current_player if hasattr(self.env, 'current_player') else None
                
                # Get the new state from the network
                new_state = {k: v.copy() if isinstance(v, dict) or isinstance(v, list) else v 
                            for k, v in self.app.network.game_state.items()}
                # Apply series overlay/state if present (from series_state messages)
                series_state = new_state.pop('series_state', None)
                
                # Extract special fields from game state
                new_current_player = new_state.pop('current_player', None)
                
                # Debug output of critical components
                print(f"Client received state update:")
                print(f"  Current player: {new_current_player}")
                print(f"  Online type: {new_state.get('online_type')}")
                print(f"  AI players: {new_state.get('ai_players', [])}")
                
                # SOLUZIONE CRUCIALE: Applica configurazione AI solo se non è già stata applicata
                if not hasattr(self, 'team_vs_ai_configured') or not self.team_vs_ai_configured:
                    ai_players = new_state.get('ai_players', [])
                    if ai_players and new_state.get('online_type') == 'team_vs_ai':
                        print("SYNC: Applicazione configurazione team_vs_ai dal server")
                        self.app.game_config['online_type'] = 'team_vs_ai'
                        
                        # Configura i giocatori AI (1 e 3)
                        for player_id in ai_players:
                            if player_id < len(self.players):
                                player = self.players[player_id]
                                player.is_human = False
                                player.is_ai = True
                                player.team_id = 1  # Team 1 (AI)
                                player.name = f"AI {player_id}"
                        
                        # Assicurati che i giocatori umani (0 e 2) siano configurati correttamente
                        for player_id in [0, 2]:
                            if player_id < len(self.players):
                                player = self.players[player_id]
                                player.is_human = True
                                player.is_ai = False
                                player.team_id = 0  # Team 0 (umano)
                                # Nomi appropriati in base all'ID locale
                                if player_id == self.local_player_id:
                                    player.name = "You"
                                else:
                                    player.name = "Partner"
                        
                        # Stampa configurazione finale
                        print("\nConfigurazione giocatori dopo sync:")
                        for player in self.players:
                            print(f"Player {player.player_id}: {player.name}, Team {player.team_id}, AI: {player.is_ai}")
                        
                        # NUOVO: Setta il flag per evitare di riapplicare la configurazione
                        self.team_vs_ai_configured = True
                        print("Configurazione team_vs_ai completata e bloccata per evitare ripetizioni")
                
                # Apply the new state to the environment (only when it contains basic fields)
                if self.env and isinstance(new_state, dict):
                    # Store hands before updating state to ensure local player's hand is preserved
                    local_hand = None
                    if self.local_player_id is not None and 'hands' in new_state:
                        if self.local_player_id in new_state['hands']:
                            local_hand = new_state['hands'][self.local_player_id]
                    
                    # Apply the new state
                    self.env.game_state = new_state
                    
                    # CRITICAL: Ensure the local player's hand is preserved and correctly updated
                    if local_hand is not None:
                        if 'hands' not in self.env.game_state:
                            self.env.game_state['hands'] = {}
                        self.env.game_state['hands'][self.local_player_id] = local_hand
                    
                    # SOLUZIONE CRUCIALE: Aggiorna il turno corrente con quello ricevuto dal server
                    if new_current_player is not None:
                        prev_player = self.env.current_player if hasattr(self.env, 'current_player') else None
                        
                        # Verifica se il giocatore è cambiato
                        player_changed = prev_player != new_current_player
                        
                        # Throttling degli aggiornamenti
                        current_time = time.time()
                        if not hasattr(self, 'last_turn_update_time'):
                            self.last_turn_update_time = 0
                        
                        # Aggiorna e stampa solo se è passato abbastanza tempo o se il giocatore è cambiato
                        if player_changed or (current_time - self.last_turn_update_time) > 0.5:  # 2 volte al secondo (0.5s)
                            print(f"SYNC: Aggiornamento turno da {prev_player} a {new_current_player}")
                            self.last_turn_update_time = current_time
                        
                        # Aggiorna sempre lo stato interno
                        self.env.current_player = new_current_player
                        self.current_player_id = new_current_player
                        
                        # Notify player if it's their turn now (questa parte rimane invariata)
                        if prev_player != new_current_player and new_current_player == self.local_player_id:
                            self.status_message = "It's your turn!"
                            self.app.resources.play_sound("card_pickup")
                            # Reset selections when it becomes the player's turn
                            self.selected_hand_card = None
                            self.selected_table_cards.clear()
                    
                    # Check for card movements and update visuals
                    if old_state and old_state != new_state:
                        # Generate animations for changes (only if both states have 'table' or 'hands')
                        self.detect_and_animate_changes(old_state, new_state, old_current_player)
                        self.waiting_for_other_player = False
                        
                        # Check for game over only if 'hands' exists in new_state
                        if 'hands' in new_state:
                            try:
                                if all(len(new_state["hands"].get(p, [])) == 0 for p in range(4)):
                                    self.game_over = True
                                    from rewards import compute_final_score_breakdown
                                    # Usa le regole correnti per il breakdown
                                    from rewards import compute_final_score_breakdown
                                    self.final_breakdown = compute_final_score_breakdown(new_state, rules=self.app.game_config.get("rules", {}))
                            except Exception:
                                pass
                    
                    # Apply series fields if provided by host
                    if series_state and isinstance(series_state, dict):
                        for key in [
                            'series_mode','series_target_points','series_num_hands','series_tiebreak',
                            'series_scores','series_hands_played','points_history','hands_won',
                            'show_intermediate_recap','last_hand_breakdown','_pending_next_starter']:
                            if key in series_state:
                                setattr(self, key, series_state[key])

                    # CRITICAL: Update player hands after applying the state
                    self.update_player_hands()

    def detect_and_animate_changes(self, old_state, new_state, old_current_player=None):
        """Improved change detection for more reliable animations"""
        # Skip if states are invalid
        if not old_state or not new_state:
            return
        
        # Get table states safely
        old_table = old_state.get('table', []) if isinstance(old_state, dict) else []
        new_table = new_state.get('table', []) if isinstance(new_state, dict) else []
        
        print(f"Client detecting changes: Old table: {old_table}")
        print(f"New table: {new_table}")
        
        # Check for last_move information which provides explicit move details
        last_move = new_state.get('last_move')
        if last_move and isinstance(last_move, dict):
            player_id = last_move.get('player')
            card_played = last_move.get('card_played')
            cards_captured = last_move.get('cards_captured')
            
            if player_id is not None and card_played:
                print(f"Detected explicit move: Player {player_id} played {card_played} and captured {cards_captured}")
                self.create_move_animations(card_played, cards_captured, player_id)
                self.app.resources.play_sound("card_play")
                return
        
        # If no explicit move info was provided, try to detect the move
        # Find cards that disappeared from the table (likely captured)
        captured_cards = []
        for card in old_table:
            if card not in new_table:
                captured_cards.append(card)
                print(f"Card {card} was likely captured from the table")
        
        # Find the card that was played (appeared on table or disappeared from hand)
        played_card = None
        player_id = None
        
        # Check each player's hand for a card that disappeared
        for p_id in range(4):
            old_hands = old_state.get('hands', {}) if isinstance(old_state, dict) else {}
            new_hands = new_state.get('hands', {}) if isinstance(new_state, dict) else {}
            old_hand = old_hands.get(p_id, [])
            new_hand = new_hands.get(p_id, [])
            
            # Skip if hands are the same or missing
            if not old_hand or not new_hand or old_hand == new_hand:
                continue
                
            print(f"Player {p_id} hand changed: {old_hand} -> {new_hand}")
            
            # Look for cards that disappeared from a hand
            for card in old_hand:
                if card not in new_hand:
                    print(f"Card {card} removed from player {p_id}'s hand")
                    
                    # This is likely the played card
                    if not played_card:
                        played_card = card
                        player_id = p_id
                        print(f"Card {card} was likely played by player {p_id}")
        
        # If we found a played card and player, create animations
        if played_card and player_id is not None:
            print(f"Creating animations for detected play: Player {player_id} played {played_card} and captured {captured_cards}")
            self.create_move_animations(played_card, captured_cards, player_id)
            self.app.resources.play_sound("card_play")
        # If only captures detected, animate them
        elif captured_cards:
            # Use most likely player (previous turn player)
            likely_player = old_current_player if old_current_player is not None else self.current_player_id
            
            # Find cards that suddenly appeared on the table (likely the played card)
            new_cards = [card for card in new_table if card not in old_table]
            played_card = new_cards[0] if new_cards else None
            
            if played_card:
                print(f"Creating animations for partial detection: Player {likely_player} played {played_card} and captured {captured_cards}")
                self.create_move_animations(played_card, captured_cards, likely_player)
            else:
                print(f"Creating animations for captures only: {captured_cards}")
                self.animate_captures_only(captured_cards, likely_player)
            
            self.app.resources.play_sound("card_play")
        # If only new cards on table, animate their appearance
        elif old_table != new_table:
            new_cards = [card for card in new_table if card not in old_table]
            if new_cards:
                print(f"Animating appearance of new cards: {new_cards}")
                for card in new_cards:
                    self.animate_card_appearance(card)
                self.app.resources.play_sound("card_play")
            else:
                print("Table changed, but could not determine specific changes")
                self.animate_table_refresh(new_table)
            
    # Add a method to animate table refresh for when specific changes can't be determined
    def animate_table_refresh(self, table_cards):
        """Animate a refresh of all table cards to ensure they're visible"""
        if not table_cards:
            return
        
        # Create a short animation for each card on the table
        # This helps ensure the client sees all cards properly
        table_center = self.table_rect.center
        
        for card in table_cards:
            # Start slightly below the table and move to center
            start_x = table_center[0] + random.randint(-20, 20)
            start_y = table_center[1] + random.randint(-20, 20)
            
            # End positions scattered around center
            end_x = table_center[0] + random.randint(-10, 10)
            end_y = table_center[1] + random.randint(-10, 10)
            
            # Create short animation
            appear_anim = CardAnimation(
                card=card,
                start_pos=(start_x, start_y),
                end_pos=(end_x, end_y),
                duration=10,
                scale_start=0.95,
                scale_end=1.0,
                rotation_start=0,
                rotation_end=0
            )
            self.animations.append(appear_anim)

    def animate_captures_only(self, captured_cards, player_id):
        """Animate cards being captured without showing a played card"""
        if not captured_cards:
            return
                
        player = self.players[player_id]
        team_id = player.team_id
        
        # Calculate pile rect for the capturing team (use new pile positions)
        pile_rect = self.get_team_pile_rect(team_id)
        
        # Calculate positions for table cards
        width = self.app.window_width
        card_width = int(width * 0.078)
        table_center = self.table_rect.center
        
        # Get current table
        table_cards = self.env.game_state["table"] if self.env else []
        
        # Create a complete list of cards to determine positions
        all_table_cards = table_cards.copy()
        # Add captured cards to the list if they're not already there
        for card in captured_cards:
            if card not in all_table_cards:
                all_table_cards.append(card)
        
        # Calculate layout for all cards
        max_spacing = self.table_rect.width * 0.8 / max(len(all_table_cards), 1)
        card_spacing = min(card_width * 1.1, max_spacing)
        start_x = self.table_rect.centerx - (len(all_table_cards) * card_spacing) // 2
        table_y = self.table_rect.centery - CARD_HEIGHT // 2
        
        # Animate each captured card with staggered timing
        for i, card in enumerate(captured_cards):
            # Try to find the correct starting position
            try:
                # Find card position in all_table_cards
                card_index = all_table_cards.index(card)
                card_x = start_x + card_index * card_spacing
                start_pos = (card_x + CARD_WIDTH // 2, table_y + CARD_HEIGHT // 2)
            except ValueError:
                # Fallback to table center if card not found
                start_pos = table_center
            
            # End near the center of the pile rect with slight variation
            end_x = pile_rect.centerx + random.randint(-6, 6)
            end_y = pile_rect.centery + random.randint(-6, 6)
            varied_end_pos = (end_x, end_y)
            
            # Create animation with staggered delay
            capture_anim = CardAnimation(
                card=card,
                start_pos=start_pos,
                end_pos=varied_end_pos,
                duration=25,
                delay=i * 5,  # Staggered delay
                scale_start=1.0,
                scale_end=0.8,
                rotation_start=0,
                rotation_end=random.randint(-10, 10)  # Slight random rotation
            )
            self.animations.append(capture_anim)
            print(f"Created capture-only animation for card {card}")

    def animate_card_appearance(self, card):
        """Animate a card appearing on the table"""
        # Create a simple animation for a card appearing on the table
        table_center = self.table_rect.center
        
        # Start slightly below the table
        start_pos = (table_center[0], table_center[1] + 50)
        
        # Animate to table center
        appear_anim = CardAnimation(
            card=card,
            start_pos=start_pos,
            end_pos=table_center,
            duration=15,
            scale_start=0.8,
            scale_end=1.0,
            rotation_start=0,
            rotation_end=0
        )
        self.animations.append(appear_anim)

    def check_game_over(self):
        """Check if the game is over"""
        if not self.env:
            return
            
        # Check if all hands are empty
        gs = getattr(self.env, 'game_state', None)
        hands = gs.get('hands', {}) if isinstance(gs, dict) else {}
        is_game_over = bool(hands) and all(len(hands.get(p, [])) == 0 for p in range(4))
        
        if is_game_over:
            self.game_over = True
            
            # Calculate final score
            from rewards import compute_final_score_breakdown
            from rewards import compute_final_score_breakdown
            self.final_breakdown = compute_final_score_breakdown(gs, rules=self.app.game_config.get("rules", {}))
            
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
            
            # Mirror drawing logic: exclude visually hidden cards
            visible_hand = hand
            if hasattr(self, 'visually_hidden_cards') and current_player.player_id in getattr(self, 'visually_hidden_cards', {}):
                hidden = self.visually_hidden_cards[current_player.player_id]
                visible_hand = [c for c in hand if c not in hidden]
            if not visible_hand:
                return None

            if is_horizontal:
                # Horizontal hand (bottom or top player)
                card_spread = card_width * 0.7
                total_width = (len(visible_hand) - 1) * card_spread + card_width
                start_x = hand_rect.centerx - total_width / 2
                
                # Y position depends on visual position
                if visual_pos == 0:  # Bottom
                    base_y = hand_rect.bottom - card_height
                else:  # Top
                    base_y = hand_rect.top
                
                # Iterate from top-most to bottom-most (reverse draw order)
                for i in range(len(visible_hand) - 1, -1, -1):
                    card = visible_hand[i]
                    # Calculate x position for this card
                    x = start_x + i * card_spread
                    # Base rect
                    card_rect = pygame.Rect(x, base_y, card_width, card_height)
                    # Apply same selection offset used in draw for accurate hit-test
                    if card == self.selected_hand_card:
                        if visual_pos == 0:
                            card_rect.move_ip(0, -15)
                        else:
                            card_rect.move_ip(0, 15)
                    # Check hit
                    if card_rect.collidepoint(pos):
                        return card
            else:
                # Vertical hand (left or right player)
                card_spread = card_height * 0.4
                total_height = (len(visible_hand) - 1) * card_spread + card_height
                start_y = hand_rect.centery - total_height / 2
                
                # X position depends on visual position
                if visual_pos == 1:  # Left
                    base_x = hand_rect.left
                else:  # Right
                    base_x = hand_rect.right - card_width
                
                # Iterate from top-most to bottom-most (reverse draw order)
                for i in range(len(visible_hand) - 1, -1, -1):
                    card = visible_hand[i]
                    # Calculate y position for this card
                    y = start_y + i * card_spread
                    # Base rect
                    card_rect = pygame.Rect(base_x, y, card_width, card_height)
                    # Apply same selection offset used in draw for accurate hit-test
                    if card == self.selected_hand_card:
                        if visual_pos == 1:
                            card_rect.move_ip(15, 0)
                        else:
                            card_rect.move_ip(-15, 0)
                    # Check hit
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
            
            # Iterate in reverse to respect draw order in case of overlap
            for i in range(len(table_cards) - 1, -1, -1):
                card = table_cards[i]
                x = start_x + i * card_spacing
                # Check if position is within this card
                card_rect = pygame.Rect(x, y, card_width, card_height)
                if card_rect.collidepoint(pos):
                    return card
            
            return None
    
    def try_make_move(self):
        """Try to make a move with the selected cards with improved handling"""
        # Debug: mostra informazioni sulla mossa
        #print(f"Tentativo di mossa - giocatore corrente: {self.current_player_id}, controllabile: {self.is_current_player_controllable()}")
        #print(f"Carta selezionata: {self.selected_hand_card}, carte tavolo: {self.selected_table_cards}")
        
        # Assicurati che sia il turno di un giocatore controllabile
        if not self.is_current_player_controllable():
            self.status_message = "Non è il tuo turno"
            return False
                
        if not self.selected_hand_card:
            self.status_message = "Select a card from your hand first"
            return False
        
        # Encode the action
        action_vec = encode_action(self.selected_hand_card, list(self.selected_table_cards))
        
        # Verifica che l'azione sia valida
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
        
        # Get the card played and cards captured for animation
        card_played, cards_captured = decode_action(valid_action)
        
        # Make the move
        try:
            # If online multiplayer, send move to server/other players
            if self.app.game_config.get("mode") == "online_multiplayer":
                self.app.network.send_move(valid_action)
                if not self.app.network.is_host:
                    # Client waits for server to update game state
                    self.waiting_for_other_player = True
                    self.status_message = "Mossa inviata, in attesa di conferma..."
                    return True
            
            # IMPORTANTE: Creiamo le animazioni che ora rimuovono automaticamente la carta dalla mano
            self.create_move_animations(card_played, cards_captured)
            
            # Play sound
            self.app.resources.play_sound("card_play")
            
            # FASE 1: Attendiamo che l'animazione della carta giocata arrivi sul tavolo
            # prima di completare l'aggiornamento dello stato del gioco
            self.waiting_for_animation = True
            self.pending_action = valid_action
            
            # Reset selection
            self.selected_hand_card = None
            self.selected_table_cards.clear()
            
            return True
        except Exception as e:
            print(f"Error making move: {e}")
            import traceback
            traceback.print_exc()
            self.status_message = "Error making move"
            return False

    
    def make_ai_move(self):
        """Make a move for the current AI player with improved animation sequencing"""
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
        
        # Print detailed information about the move
        #print(f"AI {self.current_player_id} plays {card_played} and captures {cards_captured}")
        
        try:
            # FASE 1: Create animations for the played card
            self.create_move_animations(card_played, cards_captured)
            
            # Play sound
            self.app.resources.play_sound("card_play")
            
            # FASE 2: Impostiamo l'attesa per l'animazione e salviamo l'azione da completare
            self.waiting_for_animation = True
            self.pending_action = action
            
        except Exception as e:
            print(f"Error making AI move: {e}")
            import traceback
            traceback.print_exc()
    def create_move_animations(self, card_played, cards_captured, source_player_id=None):
        """Create animations for a move with improved positioning"""
        # Get player ID - use current player if no source player specified
        player_id = source_player_id if source_player_id is not None else self.current_player_id
        current_player = self.players[player_id]
        
        # *** MODIFICA: Non modifichiamo più direttamente lo stato del gioco ***
        # Invece, memorizziamo la carta in una lista di carte visivamente nascoste
        if not hasattr(self, 'visually_hidden_cards'):
            self.visually_hidden_cards = {}
        
        # Aggiungiamo la carta alle carte nascoste di questo giocatore
        if player_id not in self.visually_hidden_cards:
            self.visually_hidden_cards[player_id] = []
        self.visually_hidden_cards[player_id].append(card_played)
        
        # NUOVO: Inizializza il set di carte in movimento se non esiste
        if not hasattr(self, 'cards_in_motion'):
            self.cards_in_motion = set()
        
        # Start position depends on player position
        if player_id == self.local_player_id:
            # Calculate position in hand
            hand_rect = current_player.hand_rect
            hand = current_player.hand_cards
            
            # Cerca la posizione della carta nella mano per un'animazione più precisa
            try:
                card_index = hand.index(card_played)
                # Calculate position with current dimensions
                width = self.app.window_width
                card_width = int(width * 0.078)
                center_x = hand_rect.centerx
                card_spread = card_width * 0.7
                
                # Calculate position
                start_x = center_x + (card_index - len(hand) // 2) * card_spread
                if self.get_visual_position(player_id) == 0:  # Bottom player
                    start_y = hand_rect.bottom - card_width * 1.5
                    start_pos = (start_x + card_width // 2, start_y + card_width * 1.5 // 2)
                else:
                    # Adjust for other positions
                    start_pos = hand_rect.center
            except ValueError:
                # Card not found in hand (fallback)
                start_pos = hand_rect.center
        else:
            # Card comes from another player's hand
            start_pos = current_player.hand_rect.center
        
        # Determina la posizione finale della carta giocata in base alle carte da catturare
        table_center = self.table_rect.center
        end_pos = table_center  # Default: centro del tavolo
        
        # Se ci sono carte da catturare, calcoliamo una posizione che si sovrappone solo parzialmente
        if cards_captured:
            # Calcola le posizioni delle carte sul tavolo
            width = self.app.window_width
            card_width = int(width * 0.078)
            card_height = int(card_width * 1.5)
            table_cards = self.env.game_state["table"] if self.env else []
            
            # Trova la posizione della prima carta da catturare
            original_table = table_cards.copy()
            for card in cards_captured:
                if card not in original_table:
                    original_table.append(card)
            
            max_spacing = self.table_rect.width * 0.8 / max(len(original_table), 1)
            card_spacing = min(card_width * 1.1, max_spacing)
            start_x = self.table_rect.centerx - (len(original_table) * card_spacing) // 2
            table_y = self.table_rect.centery - card_height // 2
            
            # Trova la carta da catturare più a sinistra o usa una posizione predefinita
            leftmost_card_index = float('inf')
            for card in cards_captured:
                try:
                    idx = original_table.index(card)
                    if idx < leftmost_card_index:
                        leftmost_card_index = idx
                except ValueError:
                    pass
            
            if leftmost_card_index != float('inf'):
                # Carta più a sinistra trovata, posiziona la carta giocata in modo che si sovrapponga parzialmente
                card_x = start_x + leftmost_card_index * card_spacing
                # Sovrapponi solo per il 50% della larghezza della carta (metà carta)
                end_pos = (card_x - card_width * 0.25, table_y + card_height / 2)
            
            #print(f"Posizione finale della carta giocata: {end_pos}")
        
        # NUOVA SEQUENZA DI ANIMAZIONE A TRE FASI
        
        # Parametri animazione
        hand_to_table_duration = 15   # Durata della prima fase (mano -> tavolo)
        plateau_duration = 30         # Durata della pausa sul tavolo
        capture_duration = 25         # Durata della fase di cattura
        inter_card_delay = 10         # Delay tra le carte nella fase di cattura
        
        # FASE 1: Animazione della carta dalla mano al tavolo
        hand_to_table = CardAnimation(
            card=card_played,
            start_pos=start_pos,
            end_pos=end_pos,
            duration=hand_to_table_duration,
            delay=0,
            scale_start=1.0,
            scale_end=1.0,
            rotation_start=0,
            rotation_end=0,
            animation_type="play"
        )
        self.animations.append(hand_to_table)
        #print(f"Creata animazione mano->tavolo per carta {card_played}")
        
        # Se ci sono carte da catturare, crea le animazioni di cattura
        if cards_captured:
            # Calculate pile rect for the capturing team (use new pile positions)
            team_id = current_player.team_id
            pile_rect = self.get_team_pile_rect(team_id)
            
            # FASE 2: NUOVO - Animazione esplicita di "plateau" per mantenere la carta visibile
            # Questa animazione mantiene la carta ferma nella stessa posizione per tutto il tempo del plateau
            plateau_anim = CardAnimation(
                card=card_played,
                start_pos=end_pos,     # Stessa posizione di fine della prima animazione
                end_pos=end_pos,       # Stessa posizione (non si muove)
                duration=plateau_duration,
                delay=hand_to_table_duration + 1,  # Avvia un frame dopo la fine della play per evitare doppio disegno
                scale_start=1.0,
                scale_end=1.0,
                rotation_start=0,
                rotation_end=0,
                animation_type="plateau"  # Nuovo tipo di animazione
            )
            self.animations.append(plateau_anim)
            #print(f"Creata animazione di plateau per carta {card_played}")
            
            # FASE 3: Animazioni dal tavolo al mazzetto
            # Il tempo totale trascorso finora è hand_to_table_duration + plateau_duration
            total_time = hand_to_table_duration + 1 + plateau_duration
            
            # Creiamo una lista di tutte le carte coinvolte nella cattura, inclusa la carta catturante
            all_capture_cards = [card_played] + list(cards_captured)
            card_width = int(self.app.window_width * 0.078)
            
            # Calcola posizioni di partenza per tutte le carte
            starting_positions = {}
            starting_positions[card_played] = end_pos  # La carta catturante parte dalla sua posizione sul tavolo
            
            # Calcola le posizioni delle carte sul tavolo
            for card in cards_captured:
                try:
                    # Find card position in the original table
                    card_index = original_table.index(card)
                    card_x = start_x + card_index * card_spacing
                    card_pos = (card_x + CARD_WIDTH // 2, table_y + CARD_HEIGHT // 2)
                    starting_positions[card] = card_pos
                except ValueError:
                    # Fallback con posizione centrale
                    starting_positions[card] = table_center
            
            # Ora creiamo le animazioni di cattura per ciascuna carta
            for i, card in enumerate(all_capture_cards):
                # Leggera variazione nella posizione finale per evitare sovrapposizione
                staggered_offset = card_width * 0.3  # Ridotto per evitare sovrapposizioni eccessive
                end_x = pile_rect.centerx + random.randint(-5, 5) + i * staggered_offset
                end_y = pile_rect.centery + random.randint(-5, 5) + i * 3
                varied_end_pos = (end_x, end_y)
                
                # Calcola il delay per questa carta
                # Base: tempo totale trascorso finora + delay incrementale
                card_delay = total_time + i * inter_card_delay
                
                #print(f"Carta {card} - delay cattura: {card_delay} frame")
                
                # NUOVO: Aggiungi la carta al set di carte in movimento quando inizia l'animazione
                # Creiamo un'animazione speciale "start_motion" che aggiungerà la carta a cards_in_motion
                motion_start_anim = CardAnimation(
                    card=card,
                    start_pos=starting_positions[card],  # Stessa posizione di partenza
                    end_pos=starting_positions[card],    # Stessa posizione (non si muove)
                    duration=1,                          # Dura solo 1 frame
                    delay=card_delay + 1,                # Un frame dopo l'inizio della capture per evitare overlap
                    scale_start=1.0,
                    scale_end=1.0,
                    rotation_start=0,
                    rotation_end=0,
                    animation_type="start_motion"        # Tipo speciale per tracciare l'inizio del movimento
                )
                self.animations.append(motion_start_anim)
                
                # Crea l'animazione con il delay calcolato
                capture_anim = CardAnimation(
                    card=card,
                    start_pos=starting_positions[card],
                    end_pos=varied_end_pos,
                    duration=capture_duration,
                    delay=card_delay + 1,  # Inizia un frame dopo il motion_start per non avere frame doppi
                    scale_start=1.0,
                    scale_end=0.8,
                    rotation_start=0,
                    rotation_end=random.randint(-10, 10),
                    animation_type="capture"
                )
                self.animations.append(capture_anim)
                #print(f"  Creata animazione tavolo->mazzetto per carta {card} con delay {card_delay}")
    
    def draw(self, surface):
        """Draw the game screen with connection loss indicators"""
        # Draw background
        surface.blit(self.app.resources.background, (0, 0))
        
        # Draw connection loss overlay and exit button if needed
        if self.connection_lost:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((self.app.window_width, self.app.window_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))  # Semi-transparent black
            surface.blit(overlay, (0, 0))
            
            # Draw exit button
            if self.exit_button:
                self.exit_button.draw(surface)
            
            # Draw reconnection status
            elapsed = time.time() - self.reconnect_start_time
            remaining = max(0, self.max_reconnection_time - elapsed)
            
            # Create animated dots
            dots = "." * ((pygame.time.get_ticks() // 500) % 4)
            
            # Draw reconnection message
            reconnect_text = f"Attempting to reconnect{dots} ({int(remaining)}s remaining)"
            text_surf = self.info_font.render(reconnect_text, True, HIGHLIGHT_RED)
            text_rect = text_surf.get_rect(
                midbottom=(self.app.window_width // 2, 
                        self.exit_button.rect.top - 10 if self.exit_button else self.app.window_height * 0.15)
            )
            surface.blit(text_surf, text_rect)
        
        # Continue with regular drawing
        # Draw table
        pygame.draw.ellipse(surface, TABLE_GREEN, 
                        self.table_rect.inflate(50, 30))
        pygame.draw.ellipse(surface, DARK_GREEN, 
                        self.table_rect.inflate(50, 30), 5)

        # Ensure elements meant to be under cards are drawn first
        # Draw capture piles (team cards) beneath any cards
        self.draw_capture_piles(surface)

        # Draw players (hands and info)
        self.draw_players(surface)

        # Draw table cards
        self.draw_table_cards(surface)

        # Draw animations; suppress normal animations while replay is active to avoid double-drawing
        if not self.replay_active:
            self.draw_animations(surface)

        # Draw replay animations if replay is active (on top of table/cards)
        if self.replay_active:
            self.draw_replay_animations(surface)

        # Draw status info
        self.draw_status_info(surface)


        # Draw live points panel next to Player 0 avatar
        self.draw_live_points_panel(surface)

        # Draw message log LAST in online multiplayer so it stays in front of team boxes and cards
        if self.app.game_config.get("mode") == "online_multiplayer":
            self.draw_message_log(surface)

        # Draw Exit button (always 'Exit')
        self.new_game_button.text = "Exit"
        self.new_game_button.draw(surface)

        # Draw confirm button if current player is controllable
        if self.env and self.is_current_player_controllable() and not self.game_over:
            self.confirm_button.draw(surface)

        # Draw replay button (always visible when not in replay mode)
        if not self.replay_active and self.env and not self.game_over:
            self.replay_button.draw(surface)

        # Intermediate recap overlay between hands (points mode) - draw on TOP of everything
        if self.show_intermediate_recap and self.last_hand_breakdown:
            self.draw_intermediate_recap(surface)

        # Draw game over screen if game is over (but not during intermediate recap)
        if self.game_over and self.final_breakdown and not self.show_intermediate_recap:
            self.draw_game_over(surface)
    
    def draw_players(self, surface):
        """Draw all players' hands and info with proper perspective handling"""
        for player in self.players:
            self.draw_player_info(surface, player)

            # Determine if player's hand should be visible
            mode = self.app.game_config.get("mode")
            is_online = mode == "online_multiplayer"
            rules = self.app.game_config.get("rules", {}) if hasattr(self.app, 'game_config') else {}
            only_turn_local = bool(rules.get("show_only_current_turn_cards", False))

            show_hand = False

            if is_online:
                # Online: only local player's hand visible
                show_hand = (player.player_id == self.local_player_id)
            elif mode == "local_multiplayer":
                # Local 4 human: either all hands or only current player's hand if option enabled
                show_hand = (player.player_id == self.current_player_id) if only_turn_local else True
            elif mode == "team_vs_ai":
                # Local team vs AI: never reveal AI hands; with the option, only reveal the current human player's hand
                if only_turn_local:
                    show_hand = (player.player_id == self.current_player_id) and (not self.players[player.player_id].is_ai)
                else:
                    show_hand = (player.player_id in [0, 2])  # Only humans in local team vs AI
            elif mode == "single_player":
                show_hand = (player.player_id == 0)

            if show_hand:
                self.draw_player_hand(surface, player)
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
        """Draw the player's hand with card faces visible and properly centered - modified to hide visually removed cards"""
        hand = player.hand_cards
        if not hand:
            return
        
        # MODIFICA: Controlla se ci sono carte da nascondere visivamente
        visually_hidden = []
        if hasattr(self, 'visually_hidden_cards') and player.player_id in self.visually_hidden_cards:
            visually_hidden = self.visually_hidden_cards[player.player_id]
        
        # Create a filtered hand without visually hidden cards
        visible_hand = [card for card in hand if card not in visually_hidden]
        
        # If no visible cards, exit early
        if not visible_hand:
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
        
        # Border parameters
        border_thickness = 1  # Sottile bordo nero
        border_radius = 8     # Angoli smussati
        
        if is_horizontal:
            # Horizontal hand layout (bottom or top)
            center_x = hand_rect.centerx
            card_spread = card_width * 0.7
            total_width = (len(visible_hand) - 1) * card_spread + card_width
            start_x = center_x - (total_width / 2)
            
            # Y position depends on whether it's top or bottom
            if visual_pos == 0:  # Bottom
                base_y = hand_rect.bottom - card_height
                rotation = 0  # No rotation for bottom player
            else:  # Top
                base_y = hand_rect.top
                rotation = 180  # Rotate 180 degrees for top player
            
            for i, card in enumerate(visible_hand):
                # Calculate x position for this card
                x = start_x + i * card_spread
                
                # Get card image
                card_img = self.app.resources.get_card_image(card)
                
                # Rotate card image
                if rotation != 0:
                    card_img = pygame.transform.rotate(card_img, rotation)
                
                # Get rect for the card
                card_rect = card_img.get_rect(center=(x + card_width/2, base_y + card_height/2))
                
                # Apply selection offset first so highlight follows moved card
                if card == self.selected_hand_card:
                    # Raise selected card for bottom player, lower for top player
                    if visual_pos == 0:
                        card_rect.move_ip(0, -15)
                    else:
                        card_rect.move_ip(0, 15)
                    # Draw selection border using moved rect
                    highlight_rect = card_rect.copy()
                    pygame.draw.rect(surface, HIGHLIGHT_BLUE, highlight_rect.inflate(10, 10), 3, border_radius=border_radius + 3)
                
                # Create a version of the card with rounded corners
                rounded_card = pygame.Surface(card_img.get_size(), pygame.SRCALPHA)
                rounded_card.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw rounded rectangle on the surface
                pygame.draw.rect(rounded_card, (255, 255, 255), rounded_card.get_rect(), 
                                border_radius=border_radius)
                
                # Use the rounded rectangle as a mask for the card
                rounded_card.blit(card_img, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                
                # Draw card with border
                pygame.draw.rect(surface, BLACK, card_rect.inflate(2, 2), border_radius=border_radius)
                surface.blit(rounded_card, card_rect)
        else:
            # Vertical hand layout (left or right)
            center_y = hand_rect.centery
            card_spread = card_height * 0.4  # Less overlap for vertical cards
            total_height = (len(visible_hand) - 1) * card_spread + card_height
            start_y = center_y - (total_height / 2)
            
            # X position depends on whether it's left or right
            if visual_pos == 1:  # Left
                base_x = hand_rect.left
                rotation = 270  # Rotate 270 degrees for left player
            else:  # Right
                base_x = hand_rect.right - card_width
                rotation = 90  # Rotate 90 degrees for right player
            
            for i, card in enumerate(visible_hand):
                # Calculate y position for this card
                y = start_y + i * card_spread
                
                # Get card image
                card_img = self.app.resources.get_card_image(card)
                
                # Rotate card image
                card_img = pygame.transform.rotate(card_img, rotation)
                
                # Get rect for the rotated card
                card_rect = card_img.get_rect(center=(base_x + card_width/2, y + card_height/2))
                
                # Apply selection offset first so highlight follows moved card
                if card == self.selected_hand_card:
                    # Move selected card outward
                    if visual_pos == 1:
                        card_rect.move_ip(15, 0)
                    else:
                        card_rect.move_ip(-15, 0)
                    # Draw selection border using moved rect
                    highlight_rect = card_rect.copy()
                    pygame.draw.rect(surface, HIGHLIGHT_BLUE, highlight_rect.inflate(10, 10), 3, border_radius=border_radius + 3)
                
                # Create a version of the card with rounded corners
                rounded_card = pygame.Surface(card_img.get_size(), pygame.SRCALPHA)
                rounded_card.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw rounded rectangle on the surface
                pygame.draw.rect(rounded_card, (255, 255, 255), rounded_card.get_rect(), 
                                border_radius=border_radius)
                
                # Use the rounded rectangle as a mask for the card
                rounded_card.blit(card_img, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                
                # Draw card with border
                pygame.draw.rect(surface, BLACK, card_rect.inflate(2, 2), border_radius=border_radius)
                surface.blit(rounded_card, card_rect)
    
    def draw_player_hidden_hand(self, surface, player):
        """Draw another player's hand with card backs rotated toward the table center"""
        hand = player.hand_cards
        if not hand:
            return
        
        # Get card back image for this player's team
        card_back = self.app.resources.get_card_back(player.team_id)
        
        # Calculate the current card width based on window size
        width = self.app.window_width
        card_width = int(width * 0.078)
        card_height = int(card_width * 1.5)
        
        # Get visual position of the player
        visual_pos = self.get_visual_position(player.player_id)
        hand_rect = player.hand_rect
        
        # Horizontal vs vertical layout
        is_horizontal = visual_pos in [0, 2]  # Bottom or top
        
        # Get table center for rotation reference
        table_center = self.table_rect.center
        
        # Border parameters
        border_thickness = 1  # Sottile bordo nero
        border_radius = 8     # Angoli smussati
        
        if is_horizontal:
            # Horizontal hand layout (bottom or top)
            card_spread = card_width * 0.7  # Cards overlap horizontally
            total_width = (len(hand) - 1) * card_spread + card_width
            start_x = hand_rect.centerx - total_width / 2
            
            # Y position depends on whether it's top or bottom
            if visual_pos == 0:  # Bottom
                base_y = hand_rect.bottom - card_height
                rotation = 0  # No rotation for bottom player
            else:  # Top
                base_y = hand_rect.top
                rotation = 180  # Rotate 180 degrees for top player
            
            for i in range(len(hand)):
                # Calculate x position for this card
                x = start_x + i * card_spread
                
                # Rotate card back appropriately
                rotated_back = pygame.transform.rotate(card_back, rotation)
                
                # Get rect for the rotated card
                card_rect = rotated_back.get_rect(center=(x + card_width/2, base_y + card_height/2))
                
                # Create a version of the card with rounded corners
                rounded_card = pygame.Surface(rotated_back.get_size(), pygame.SRCALPHA)
                rounded_card.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw rounded rectangle on the surface
                pygame.draw.rect(rounded_card, (255, 255, 255), rounded_card.get_rect(), 
                                border_radius=border_radius)
                
                # Use the rounded rectangle as a mask for the card
                rounded_card.blit(rotated_back, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                
                # Draw card with border
                pygame.draw.rect(surface, BLACK, card_rect.inflate(2, 2), border_radius=border_radius)
                surface.blit(rounded_card, card_rect)
        else:
            # Vertical hand layout (left or right)
            card_spread = card_height * 0.4  # Less overlap for vertical cards
            total_height = (len(hand) - 1) * card_spread + card_height
            start_y = hand_rect.centery - total_height / 2
            
            # X position depends on whether it's left or right
            if visual_pos == 1:  # Left
                base_x = hand_rect.left
                rotation = 270  # Rotate 270 degrees for left player
            else:  # Right
                base_x = hand_rect.right - card_width
                rotation = 90  # Rotate 90 degrees for right player
            
            for i in range(len(hand)):
                # Calculate y position for this card
                y = start_y + i * card_spread
                
                # Rotate card back appropriately
                rotated_back = pygame.transform.rotate(card_back, rotation)
                
                # Get rect for the rotated card
                card_rect = rotated_back.get_rect(center=(base_x + card_width/2, y + card_height/2))
                
                # Create a version of the card with rounded corners
                rounded_card = pygame.Surface(rotated_back.get_size(), pygame.SRCALPHA)
                rounded_card.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw rounded rectangle on the surface
                pygame.draw.rect(rounded_card, (255, 255, 255), rounded_card.get_rect(), 
                                border_radius=border_radius)
                
                # Use the rounded rectangle as a mask for the card
                rounded_card.blit(rotated_back, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                
                # Draw card with border
                pygame.draw.rect(surface, BLACK, card_rect.inflate(2, 2), border_radius=border_radius)
                surface.blit(rounded_card, card_rect)
    
    def draw_table_cards(self, surface):
        """Draw the cards on the table without any rotation but with rounded corners and border"""
        if not self.env:
            return
        
        # Use replay table state if replay is active, otherwise use current table state
        if self.replay_active:
            # Durante il replay usiamo SEMPRE lo stato del replay, anche se vuoto
            table_cards = self.replay_table_state
        else:
            table_cards = self.env.game_state["table"]
            
        # Se lo stato segnala tavolo vuoto ma ci sono animazioni di "play" in corso che dovrebbero
        # aver già portato carte sul tavolo, evita di mostrare "No cards on table" e lascia lo spazio
        # libero per l'animazione, così l'utente non crede che manchino le carte.
        if not table_cards:
            # Se c'è un'animazione in corso che non sia di cattura (plateau/play), non mostrare il testo.
            if any(getattr(anim, 'animation_type', '') in ('play', 'plateau', 'replay_play') and not getattr(anim, 'done', False)
                   for anim in getattr(self, 'animations', [])):
                return
            # Draw "No cards on table" text
            text_surf = self.info_font.render("No cards on table", True, WHITE)
            text_rect = text_surf.get_rect(center=self.table_rect.center)
            surface.blit(text_surf, text_rect)
            return
        
        # NUOVO: Assicurati che esista il set di carte in movimento
        if not hasattr(self, 'cards_in_motion'):
            self.cards_in_motion = set()
        
        # NUOVO: Assicurati che esista il set di carte in movimento per il replay
        if not hasattr(self, 'replay_cards_in_motion'):
            self.replay_cards_in_motion = set()
        
        # Get the current card size based on window size
        width = self.app.window_width
        card_width = int(width * 0.078)
        card_height = int(card_width * 1.5)
        
        # Border parameters
        border_thickness = 1  # Sottile bordo nero
        border_radius = 8     # Angoli smussati
        
        # Calculate positions for table layout - use table width to determine spacing
        max_spacing = self.table_rect.width * 0.8 / max(len(table_cards), 1)
        card_spacing = min(card_width * 1.1, max_spacing)
        start_x = self.table_rect.centerx - (len(table_cards) * card_spacing) // 2
        y = self.table_rect.centery - card_height // 2
        
        for i, card in enumerate(table_cards):
            # NUOVO: Salta la carta solo se è effettivamente in animazione attiva.
            # Se la carta è nel set ma non ha un'animazione in corso (set stantio), disegna comunque e ripulisci.
            in_motion_flag = card in self.cards_in_motion or (self.replay_active and card in self.replay_cards_in_motion)
            if in_motion_flag:
                # Se è in motion, evita di disegnarla se esiste un'animazione attiva (normale o replay)
                has_active_anim = any(getattr(anim, 'card', None) == card and not getattr(anim, 'done', False) for anim in self.animations)
                if self.replay_active and not has_active_anim:
                    has_active_anim = any(getattr(anim, 'card', None) == card and not getattr(anim, 'done', False) for anim in self.replay_animations)
                if has_active_anim:
                    continue
                # Ripulisci flag stantio se nessuna animazione è attiva per questa carta
                if card in self.cards_in_motion:
                    try:
                        self.cards_in_motion.remove(card)
                    except KeyError:
                        pass
                if self.replay_active and card in getattr(self, 'replay_cards_in_motion', set()):
                    try:
                        self.replay_cards_in_motion.remove(card)
                    except KeyError:
                        pass
                
            x = start_x + i * card_spacing
            
            # Get card image
            card_img = self.app.resources.get_card_image(card)
            
            # Create rect for the card at its position (no rotation)
            card_rect = pygame.Rect(x, y, card_width, card_height)
            
            # Highlight selected table cards
            if card in self.selected_table_cards:
                pygame.draw.rect(surface, HIGHLIGHT_RED, card_rect.inflate(10, 10), 3, border_radius=border_radius + 3)
            
            # Create a version of the card with rounded corners
            rounded_card = pygame.Surface(card_img.get_size(), pygame.SRCALPHA)
            rounded_card.fill((0, 0, 0, 0))  # Transparent background
            
            # Draw rounded rectangle on the surface
            pygame.draw.rect(rounded_card, (255, 255, 255), rounded_card.get_rect(), 
                            border_radius=border_radius)
            
            # Use the rounded rectangle as a mask for the card
            rounded_card.blit(card_img, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            
            # Draw card with border
            pygame.draw.rect(surface, BLACK, card_rect.inflate(2, 2), border_radius=border_radius)
            surface.blit(rounded_card, card_rect)
    def draw_capture_piles(self, surface):
        """Draw realistic capture piles for both teams with scopa highlights.

        - Cards in piles are rendered as real face-down cards, slightly misaligned
          to convey stack height.
        - For each scopa, the played card is left inside the pile, rotated about 90° and
          face-up, anchored to the center of the left long side of the covered card
          directly below the face-up card (or the first covered card if none below).
        - Cards captured in a scopa remain covered
        - If there's more than one scopa (so more than one card face up) they are distanced by 40° each.
        - Captured cards are drawn at ~1/3 the size of hand cards.
        """
        if not self.env:
            return
                
        gs = self.env.game_state
        captured = gs["captured_squads"]
        width = self.app.window_width
        height = self.app.window_height
        
        # Mini card size ~ 1/3 of hand card size
        mini_w = max(1, int(CARD_WIDTH * 0.33))
        mini_h = max(1, int(CARD_HEIGHT * 0.33))

        # Slight misalignment per card (very small, to avoid clutter)
        tiny_dx = max(1, int(mini_w * 0.06))
        tiny_dy = max(1, int(mini_h * 0.06))

        padding = int(min(width, height) * 0.008)

        def team_of_player(player_id: int) -> int:
            return 0 if player_id in [0, 2] else 1

        # Collect capture moves per team to reconstruct pile order
        team_moves = {0: [], 1: []}
        for move in gs.get("history", []):
            ctype = move.get("capture_type")
            if ctype in ("capture", "scopa"):
                t = team_of_player(move.get("player"))
                team_moves[t].append(move)

        def blit_rounded(surface_ref, img, dest_rect, border_radius=6, rotation_degrees=0):
            # Render a rounded, optionally rotated card with border drawn on the card surface
            if rotation_degrees:
                img = pygame.transform.rotate(img, rotation_degrees)
            rounded_card = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            rounded_card.fill((0, 0, 0, 0))
            inner_rect = rounded_card.get_rect()
            pygame.draw.rect(rounded_card, (255, 255, 255), inner_rect, border_radius=border_radius)
            rounded_card.blit(img, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            pygame.draw.rect(rounded_card, BLACK, inner_rect, width=2, border_radius=border_radius)
            # Recompute destination if image has been rotated (size may differ)
            if rotation_degrees:
                dest_rect = rounded_card.get_rect(center=dest_rect.center)
            surface_ref.blit(rounded_card, dest_rect)

        def draw_team_pile(team_id: int):
            rect = self.get_team_pile_rect(team_id)

            # Background panel for the pile area and label
            panel_color = LIGHT_BLUE if team_id == 0 else HIGHLIGHT_RED
            pygame.draw.rect(surface, panel_color, rect, border_radius=5)

            count = len(captured[team_id])
            text = f"Team {team_id}: {count} cards"
            text_surf = self.small_font.render(text, True, WHITE)
            surface.blit(text_surf, (rect.left + 10, rect.top + 5))

            # Pre-scale card back for this team
            back_img = self.app.resources.get_card_back(team_id)
            scaled_back = pygame.transform.scale(back_img, (mini_w, mini_h))

            # Starting position for top of the pile
            cur_x = rect.left + padding
            cur_y = rect.top + padding + self.small_font.get_height() + 6  # leave space for label

            drawn_cards = set()

            # Number and index of scopa moves to distribute face-up angles (40° increments)
            num_scopa = sum(1 for m in team_moves[team_id] if m.get("capture_type") == "scopa")
            scopa_idx_seen = 0

            # Draw captures in chronological order to preserve stack order
            for move in team_moves[team_id]:
                ctype = move.get("capture_type")
                captured_cards = move.get("captured_cards") or []
                played_card = move.get("played_card")

                if ctype == "scopa":
                    # Base position at current top of pile; face-up will anchor to the covered card just below it
                    base_x = cur_x
                    base_y = cur_y

                    # Angle distribution among all scopa face-up cards for this team (±40° steps around 90°)
                    angle_offset = 0.0
                    if num_scopa > 1:
                        start = -40.0 * (num_scopa - 1) / 2.0
                        angle_offset = start + scopa_idx_seen * 40.0

                    # Played card shown face-up (drawn first so it stays UNDER captured cards), rotated 90° + offset,
                    # anchored to the covered card just below it, then detached to their vertical axis by half length
                    if played_card:
                        face_img = self.app.resources.get_card_image(played_card)
                        if face_img:
                            scaled_face = pygame.transform.scale(face_img, (mini_w, mini_h))
                            # Compose rounded card with border BEFORE rotation so the border rotates with the card
                            base_card = pygame.Surface((mini_w, mini_h), pygame.SRCALPHA)
                            base_card.fill((0, 0, 0, 0))
                            base_rect = base_card.get_rect()
                            pygame.draw.rect(base_card, (255, 255, 255), base_rect, border_radius=6)
                            base_card.blit(scaled_face, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                            pygame.draw.rect(base_card, BLACK, base_rect, width=2, border_radius=6)
                            # Desired rotation and precise anchoring with pivot at TOP short side center
                            angle_deg = -90 + angle_offset
                            rotated_card = pygame.transform.rotate(base_card, -angle_deg)

                            # Determine the anchor on the covered card just below the face-up card
                            covered_below_index = max(0, len(captured_cards) - 1)
                            anchor_x = base_x - covered_below_index * tiny_dx
                            anchor_y = base_y + covered_below_index * tiny_dy
                            anchor = pygame.math.Vector2(anchor_x, anchor_y + mini_h / 2.0)

                            # Pivot math: rotate around the center of the TOP short side of the unrotated card
                            center_local = pygame.math.Vector2(mini_w / 2.0, mini_h / 2.0)
                            pivot_local = pygame.math.Vector2(mini_w / 2.0, 0.0)  # top middle
                            v = pivot_local - center_local
                            v_rot = v.rotate(angle_deg)

                            # Place rotated surface so that rotated pivot lands on the anchor
                            dest = rotated_card.get_rect()
                            dest.center = (anchor.x - v_rot.x, anchor.y - v_rot.y)
                            
                            # Detach along the card's long axis by half the card length
                            shift_dir = pygame.math.Vector2(0.0, 1.0).rotate(angle_deg)
                            dest.x += int(shift_dir.x * (mini_h * 0.5))
                            dest.y += int(shift_dir.y * (mini_h * 0.5))
                            
                            
                            surface.blit(rotated_card, dest)
                            drawn_cards.add(played_card)
                            scopa_idx_seen += 1

                    # Now lay out captured cards strongly offset to the left, kept covered (drawn after => on top)
                    for i, card in enumerate(captured_cards):
                        x = base_x - i * tiny_dx
                        y = base_y + i * tiny_dy
                        dest = pygame.Rect(x, y, mini_w, mini_h)
                        blit_rounded(surface, scaled_back, dest)
                        drawn_cards.add(card)

                    # Advance the pile slightly for next additions (keep the pile compact)
                    cur_x += tiny_dx
                    cur_y += tiny_dy
                else:
                    # Normal capture: all cards remain covered, slightly misaligned
                    for card in captured_cards:
                        dest = pygame.Rect(cur_x, cur_y, mini_w, mini_h)
                        blit_rounded(surface, scaled_back, dest)
                        drawn_cards.add(card)
                        cur_x += tiny_dx
                        cur_y += tiny_dy

                    # Add the played card on top, still covered
                    if played_card:
                        dest = pygame.Rect(cur_x, cur_y, mini_w, mini_h)
                        blit_rounded(surface, scaled_back, dest)
                        drawn_cards.add(played_card)
                        cur_x += tiny_dx
                        cur_y += tiny_dy

            # Draw any leftover captured cards not represented in history (e.g., end-of-round sweep)
            for card in captured[team_id]:
                if card not in drawn_cards:
                    dest = pygame.Rect(cur_x, cur_y, mini_w, mini_h)
                    blit_rounded(surface, scaled_back, dest)
                    cur_x += tiny_dx
                    cur_y += tiny_dy

        # Draw both teams' piles
        draw_team_pile(0)
        draw_team_pile(1)
    
    def draw_animations(self, surface):
        """Draw all active card animations with rounded corners and borders"""
        # Border parameters
        border_thickness = 1  # Sottile bordo nero
        border_radius = 8     # Angoli smussati
        
        for anim in self.animations:
            if anim.current_frame < 0:
                continue  # Animation in delay phase
                
            # Get card image
            card_img = self.app.resources.get_card_image(anim.card)
            
            # Apply scaling
            scale = anim.get_current_scale()
            scaled_width = int(CARD_WIDTH * scale)
            scaled_height = int(CARD_HEIGHT * scale)
            scaled_img = pygame.transform.scale(card_img, (scaled_width, scaled_height))
            
            # Apply rotation
            rotation = anim.get_current_rotation()
            # Compose rounded card with border BEFORE rotation so border/rounding rotate with the card
            base_card = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
            base_card.fill((0, 0, 0, 0))
            base_rect = base_card.get_rect()
            pygame.draw.rect(base_card, (255, 255, 255), base_rect, border_radius=border_radius)
            base_card.blit(scaled_img, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            pygame.draw.rect(base_card, BLACK, base_rect, width=2, border_radius=border_radius)
            rotated_img = pygame.transform.rotate(base_card, rotation)
            
            # Get current position
            pos = anim.get_current_pos()
            
            # Create rect centered at position
            rect = rotated_img.get_rect(center=pos)
            
            # Draw rotated card surface directly (already has rounded corners and border)
            surface.blit(rotated_img, rect)
    
    def draw_status_info(self, surface):
        """Draw game status information with emphasis on player turn"""
        width = self.app.window_width
        height = self.app.window_height

        # Anchor for top-left placement under the Exit button
        if hasattr(self, "new_game_button") and self.new_game_button:
            anchor_left = self.new_game_button.rect.left
            anchor_top = self.new_game_button.rect.bottom + int(height * 0.01)
        else:
            anchor_left = int(width * 0.01)
            anchor_top = int(height * 0.08)
        current_y = anchor_top
        drew_gold_line = False

        # Draw status message
        if self.status_message:
            is_turn_msg = "your turn" in self.status_message.lower()
            msg_color = GOLD if is_turn_msg else WHITE
            msg_font = self.title_font if is_turn_msg else self.info_font

            msg_surf = msg_font.render(self.status_message, True, msg_color)
            if is_turn_msg:
                # Place gold turn message under Exit, top-left
                msg_rect = msg_surf.get_rect(topleft=(anchor_left, current_y))
                surface.blit(msg_surf, msg_rect)
                current_y = msg_rect.bottom + int(height * 0.008)
                drew_gold_line = True
            else:
                # Keep non-highlighted messages centered at the top
                msg_rect = msg_surf.get_rect(center=(width // 2, int(height * 0.026)))
                surface.blit(msg_surf, msg_rect)

        # Draw current player indicator
        if self.env:
            current_player = self.players[self.current_player_id]
            turn_text = f"Current turn: {current_player.name} (Team {current_player.team_id})"

            # Always show current turn in gold at top-left under Exit
            is_local_turn = self.current_player_id == self.local_player_id
            turn_color = GOLD

            turn_surf = self.info_font.render(turn_text, True, turn_color)
            turn_rect = turn_surf.get_rect(topleft=(anchor_left, current_y))
            surface.blit(turn_surf, turn_rect)
            current_y = turn_rect.bottom + int(height * 0.006)

            # Extra gold indicator for local player's turn → move under Exit
            if is_local_turn:
                indicator_text = "YOUR TURN"
                indicator_surf = self.info_font.render(indicator_text, True, GOLD)
                indicator_rect = indicator_surf.get_rect(topleft=(anchor_left, current_y))
                surface.blit(indicator_surf, indicator_rect)
                current_y = indicator_rect.bottom + int(height * 0.006)

            # Draw difficulty info if playing against AI (all online modes with AI)
            online_type = self.app.game_config.get("online_type")
            if (self.app.game_config.get("mode") == "single_player" or
                online_type in ("team_vs_ai", "humans_plus_ai", "three_humans_one_ai")):
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
                diff_rect = diff_surf.get_rect(topright=(width - int(width * 0.02), int(height * 0.026)))
                surface.blit(diff_surf, diff_rect)

            # Live series recap (points/hands)
            if self.series_mode == "points":
                hist_text = ", ".join([f"{a}-{b}" for (a, b) in self.points_history]) if self.points_history else "-"
                tot0 = sum(x for x, _ in self.points_history)
                tot1 = sum(x for _, x in self.points_history)
                tot_text = f"Tot: {tot0}-{tot1} / Target {self.series_target_points}"
                hist_surf = self.small_font.render(f"Storico mani (T0-T1): {hist_text}", True, WHITE)
                tot_surf = self.small_font.render(tot_text, True, GOLD)
                surface.blit(hist_surf, (anchor_left, current_y))
                current_y += int(height * 0.022)
                surface.blit(tot_surf, (anchor_left, current_y))
                current_y += int(height * 0.024)
            elif self.series_mode == "hands":
                won_text = f"Mani vinte: {self.hands_won[0]}-{self.hands_won[1]} / {self.series_num_hands}"
                won_surf = self.small_font.render(won_text, True, GOLD)
                surface.blit(won_surf, (anchor_left, current_y))
                current_y += int(height * 0.024)

        # Small rules summary box under the turn info (top-left)
        rules = self.app.game_config.get("rules", {})
        if rules:
            # Compose compact lines
            mode_type = rules.get("mode_type", "oneshot")
            lines = []
            if mode_type == "points":
                lines.append(f"Modalità: Punti ({int(rules.get('target_points', 21))})")
            elif mode_type == "hands":
                lines.append(f"Modalità: Mani ({int(rules.get('num_hands', 1))})")
            else:
                lines.append("Modalità: One-shot")

            st = rules.get("starting_team", "random")
            lines.append(f"Chi inizia: {st}")

            # Per modalità a mani, mostra anche mani vinte
            if mode_type == "hands":
                try:
                    wins0, wins1 = self.hands_won
                    total_hands = int(self.series_num_hands)
                    lines.append(f"Mani vinte: {wins0}-{wins1} / {total_hands}")
                except Exception:
                    pass

            # Active toggles
            if rules.get("asso_piglia_tutto", False):
                ap_line = "Asso piglia tutto"
                if rules.get("scopa_on_asso_piglia_tutto", False):
                    ap_line += " + Scopa"
                if rules.get("asso_piglia_tutto_posabile", False):
                    if rules.get("asso_piglia_tutto_posabile_only_empty", False):
                        ap_line += " (posabile: solo tavolo vuoto)"
                    else:
                        ap_line += " (posabile: sempre)"
                lines.append(ap_line)
            if rules.get("scopa_on_last_capture", False):
                lines.append("Scopa su ultima")
            if rules.get("re_bello", False):
                lines.append("Re Bello")
            if rules.get("napola", False):
                ns = rules.get("napola_scoring", "fixed3")
                lines.append("Napola (" + ("len" if ns == "length" else "3") + ")")
            lim = rules.get("max_consecutive_scope")
            if lim is not None:
                lines.append(f"Limite scope: {int(lim)}")
            if not rules.get("last_cards_to_dealer", True):
                lines.append("Ultime al team dell'ultima presa: OFF")
            # Tempo per mossa AI rimosso

            # Box geometry
            pad_x = int(width * 0.008)
            pad_y = int(height * 0.006)
            box_w = int(width * 0.24)
            # Measure height from lines
            line_h = self.small_font.get_height()
            box_h = pad_y * 2 + line_h * len(lines)
            box_rect = pygame.Rect(anchor_left, current_y + int(height * 0.006), box_w, box_h)
            # Background
            box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(box_surf, (0, 0, 0, 150), box_surf.get_rect(), border_radius=8)
            pygame.draw.rect(box_surf, LIGHT_BLUE, box_surf.get_rect(), 2, border_radius=8)
            # Render lines
            y = pad_y
            for text in lines[:8]:  # cap length
                ts = self.small_font.render(text, True, WHITE)
                box_surf.blit(ts, (pad_x, y))
                y += line_h
            surface.blit(box_surf, box_rect)
            current_y = box_rect.bottom + int(height * 0.008)

            # In modalità a punti, mostra lo storico e la somma cumulativa
            if mode_type == "points" and len(self.points_history) > 0:
                hist_pad_x = pad_x
                hist_pad_y = pad_y
                # Somme cumulative
                tot0 = sum(p0 for p0, _ in self.points_history)
                tot1 = sum(p1 for _, p1 in self.points_history)
                title = self.small_font.render(f"Storico mani (T0-T1)  Tot: {tot0}-{tot1}", True, GOLD)
                # Calcola larghezza dinamica in base al numero di voci
                entries = [f"{p0}-{p1}" for (p0, p1) in self.points_history[-8:]]
                hist_lines = [", ".join(entries[i:i+4]) for i in range(0, len(entries), 4)]
                hist_line_h = self.small_font.get_height()
                hist_box_w = box_w
                hist_box_h = hist_pad_y * 2 + hist_line_h * (1 + len(hist_lines))
                hist_rect = pygame.Rect(anchor_left, current_y, hist_box_w, hist_box_h)
                hist_surf = pygame.Surface((hist_box_w, hist_box_h), pygame.SRCALPHA)
                pygame.draw.rect(hist_surf, (0, 0, 0, 150), hist_surf.get_rect(), border_radius=8)
                pygame.draw.rect(hist_surf, LIGHT_BLUE, hist_surf.get_rect(), 2, border_radius=8)
                # Title
                hist_surf.blit(title, (hist_pad_x, hist_pad_y))
                y = hist_pad_y + hist_line_h
                for row in hist_lines:
                    ts = self.small_font.render(row, True, WHITE)
                    hist_surf.blit(ts, (hist_pad_x, y))
                    y += hist_line_h
                surface.blit(hist_surf, hist_rect)
    
    def draw_live_points_panel(self, surface):
        """Small live score panel near Player 0's avatar showing current statuses.

        Displays per team:
        - Carte (majority so far)
        - Ori (denari majority)
        - Primiera leader
        - Settebello possession
        - Scope count
        """
        if not getattr(self, 'env', None) or not getattr(self, 'players', None):
            return

        # Anchor to player 0 avatar
        try:
            player0 = next(p for p in self.players if p.player_id == 0)
        except StopIteration:
            return

        width = self.app.window_width
        height = self.app.window_height
        anchor = player0.avatar_rect

        panel_w = int(width * 0.16)
        panel_h = int(height * 0.08)
        margin = max(6, int(min(width, height) * 0.01))

        # Prefer to the right of avatar; fallback left if needed
        x = anchor.right + margin
        y = anchor.top
        if x + panel_w > width - margin:
            x = anchor.left - margin - panel_w
        x = max(margin, min(x, width - margin - panel_w))
        y = max(margin, min(y, height - margin - panel_h))

        panel_rect = pygame.Rect(x, y, panel_w, panel_h)

        # Semi-transparent rounded background + border
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (0, 0, 0, 170), panel_surf.get_rect(), border_radius=8)
        pygame.draw.rect(panel_surf, WHITE, panel_surf.get_rect(), 2, border_radius=8)

        small_font = pygame.font.SysFont(None, int(height * 0.022))
        # Game rules for conditional metrics
        rules = self.app.game_config.get("rules", {})

        # Gather game data
        gs = self.env.game_state
        captured = gs.get("captured_squads", {0: [], 1: []})
        team_cards_0 = captured[0] if isinstance(captured, (list, tuple)) else captured.get(0, [])
        team_cards_1 = captured[1] if isinstance(captured, (list, tuple)) else captured.get(1, [])

        # Remaining cards pool (all hands + table)
        remaining_cards = []
        hands = gs.get("hands", {})
        for pid in range(4):
            remaining_cards.extend(hands.get(pid, []))
        remaining_cards.extend(gs.get("table", []))
        rem_total = len(remaining_cards)
        rem_den = sum(1 for r, s in remaining_cards if s == 'denari')

        # Scope counts
        scope0 = 0
        scope1 = 0
        for move in gs.get("history", []):
            if move.get("capture_type") == "scopa":
                if move.get("player") in [0, 2]:
                    scope0 += 1
                else:
                    scope1 += 1

        # Carte majority indicator
        c0 = len(team_cards_0)
        c1 = len(team_cards_1)
        lead_c0 = 1 if c0 > c1 else 0 if c0 == c1 else -1

        # Denari majority
        den0 = sum(1 for r, s in team_cards_0 if s == 'denari')
        den1 = sum(1 for r, s in team_cards_1 if s == 'denari')
        lead_d0 = 1 if den0 > den1 else 0 if den0 == den1 else -1

        # Primiera leader
        val_map = {1: 16, 2: 12, 3: 13, 4: 14, 5: 15, 6: 18, 7: 21, 8: 10, 9: 10, 10: 10}
        def prim_sum(cards):
            best = {"denari": 0, "coppe": 0, "spade": 0, "bastoni": 0}
            for (rank, suit) in cards:
                v = val_map.get(rank, 0)
                if v > best.get(suit, 0):
                    best[suit] = v
            return sum(best.values())

        prim0 = prim_sum(team_cards_0)
        prim1 = prim_sum(team_cards_1)
        lead_p0 = 1 if prim0 > prim1 else 0 if prim0 == prim1 else -1

        # Settebello possession
        sb0 = (7, 'denari') in team_cards_0
        sb1 = (7, 'denari') in team_cards_1

        # Render two compact rows: team 0 and 1
        row_y = 6
        row_h = (panel_h - 12) // 2

        def draw_team_row(team_id):
            nonlocal row_y
            color = LIGHT_BLUE if team_id == 0 else HIGHLIGHT_RED
            if team_id == 0:
                carte_mark = "✓" if lead_c0 == 1 else "=" if lead_c0 == 0 else ""
                denari_mark = "✓" if lead_d0 == 1 else "=" if lead_d0 == 0 else ""
                prim_mark = "✓" if lead_p0 == 1 else "=" if lead_p0 == 0 else ""
                sette_mark = "✓" if sb0 else ""
                scope_text = str(scope0)
            else:
                carte_mark = "✓" if lead_c0 == -1 else "=" if lead_c0 == 0 else ""
                denari_mark = "✓" if lead_d0 == -1 else "=" if lead_d0 == 0 else ""
                prim_mark = "✓" if lead_p0 == -1 else "=" if lead_p0 == 0 else ""
                sette_mark = "✓" if sb1 else ""
                scope_text = str(scope1)

            # Team label compact
            team_label = f"T{team_id}"
            team_surf = small_font.render(team_label, True, color)
            panel_surf.blit(team_surf, (8, row_y))

            # Helper to map a metric state to a colored dot per requirement
            def dot_color(win, lose, tie, advantage):
                # win: already won definitively; lose: already lost definitively
                # tie: exactly equal; advantage: currently in advantage (not final)
                if lose:
                    return RED
                if win:
                    return GREEN
                if tie:
                    return YELLOW
                if advantage > 0:
                    return LIGHT_GREEN
                if advantage < 0:
                    return ORANGE
                return GRAY

            # Determine statuses for each metric
            # Live definitiveness per requirement: assume opponent takes all remaining cards
            # Carte/Denari/Primiera: if opponent can't overturn even with all remaining → win; viceversa → lose
            # Settebello: definitive when captured
            # Compute advantages from team 0 perspective; invert for team 1 when drawing
            adv_c = (c0 - c1)
            adv_d = (den0 - den1)
            adv_p = (prim0 - prim1)

            if team_id == 1:
                adv_c *= -1
                adv_d *= -1
                adv_p *= -1

            # Carte (cards majority) live definitiveness
            if team_id == 0:
                carte_win = c0 > c1 + rem_total
                carte_lose = c0 + rem_total < c1
            else:
                carte_win = c1 > c0 + rem_total
                carte_lose = c1 + rem_total < c0
            carte_tie = (c0 == c1)
            # Denari (ori) live definitiveness
            if team_id == 0:
                den_win = den0 > den1 + rem_den
                den_lose = den0 + rem_den < den1
            else:
                den_win = den1 > den0 + rem_den
                den_lose = den1 + rem_den < den0
            den_tie = (den0 == den1)
            # Primiera live definitiveness via maximum potential with remaining cards
            if team_id == 0:
                opp_max = prim_sum(team_cards_1 + remaining_cards)
                team_max = prim_sum(team_cards_0 + remaining_cards)
                prim_win = prim0 > opp_max
                prim_lose = team_max < prim1
            else:
                opp_max = prim_sum(team_cards_0 + remaining_cards)
                team_max = prim_sum(team_cards_1 + remaining_cards)
                prim_win = prim1 > opp_max
                prim_lose = team_max < prim0
            prim_tie = (prim0 == prim1)
            # Settebello
            sette_win = sb0 if team_id == 0 else sb1
            sette_lose = (sb1 if team_id == 0 else sb0)
            sette_tie = (not sb0) and (not sb1)

            # Render dots line: C O P 7 S
            cx = 8 + team_surf.get_width() + 8
            cy = row_y + small_font.get_height() // 2
            radius = max(4, small_font.get_height() // 4)

            def draw_dot(label, color_val):
                nonlocal cx
                lbl = small_font.render(label, True, WHITE)
                panel_surf.blit(lbl, (cx, row_y))
                cx += lbl.get_width() + 4
                pygame.draw.circle(panel_surf, color_val, (cx + radius, cy), radius)
                pygame.draw.circle(panel_surf, WHITE, (cx + radius, cy), radius, 1)
                cx += radius * 2 + 8

            draw_dot("C", dot_color(carte_win, carte_lose, carte_tie, adv_c))
            draw_dot("O", dot_color(den_win, den_lose, den_tie, adv_d))
            draw_dot("P", dot_color(prim_win, prim_lose, prim_tie, adv_p))
            draw_dot("7", dot_color(sette_win, sette_lose, sette_tie, 0))

            # Re Bello (King of denari) indicator if active
            if rules.get("re_bello", False):
                rb_team_has = (10, 'denari') in (team_cards_0 if team_id == 0 else team_cards_1)
                rb_opp_has = (10, 'denari') in (team_cards_1 if team_id == 0 else team_cards_0)
                rb_tie = (not rb_team_has) and (not rb_opp_has)
                draw_dot("RB", dot_color(rb_team_has, rb_opp_has, rb_tie, 0))

            # Cavallina (Napola progress) if active
            if rules.get("napola", False):
                team_cards = team_cards_0 if team_id == 0 else team_cards_1
                opp_cards = team_cards_1 if team_id == 0 else team_cards_0
                team_den_ranks = {r for (r, s) in team_cards if s == 'denari'}
                opp_den_ranks = {r for (r, s) in opp_cards if s == 'denari'}

                have_all3 = {1, 2, 3}.issubset(team_den_ranks)
                opp_has_any_123 = len(opp_den_ranks.intersection({1, 2, 3})) > 0
                team_has_any_123 = len(team_den_ranks.intersection({1, 2, 3})) > 0

                if have_all3:
                    # Compute cavallina length from Ace upward
                    length = 0
                    r = 1
                    while r in team_den_ranks:
                        length += 1
                        r += 1
                    # Render like scope: label + green number
                    cav_lbl = small_font.render("N", True, WHITE)
                    panel_surf.blit(cav_lbl, (cx, row_y))
                    cx += cav_lbl.get_width() + 4
                    cav_text = small_font.render(str(length), True, LIGHT_GREEN)
                    panel_surf.blit(cav_text, (cx, row_y))
                    cx += cav_text.get_width() + 8
                else:
                    # Status dot: lost if opponent holds any of {A,2,3};
                    # tie if none holds; advantage if team holds some and opponent none.
                    cav_lose = opp_has_any_123 or (team_has_any_123 and len(opp_den_ranks.intersection({1, 2, 3})) > 0)
                    cav_win = False  # not definitive until have_all3
                    cav_tie = (not team_has_any_123) and (not opp_has_any_123)
                    advantage = 1 if (team_has_any_123 and not opp_has_any_123) else 0
                    draw_dot("N", dot_color(cav_win, cav_lose, cav_tie, advantage))

            # Scope as count with color reflecting advantage
            scope_team = scope0 if team_id == 0 else scope1
            scope_other = scope1 if team_id == 0 else scope0
            scope_adv = scope_team - scope_other
            scope_color = LIGHT_GREEN if scope_adv > 0 else YELLOW if scope_adv == 0 else ORANGE
            scope_lbl = small_font.render("S", True, WHITE)
            panel_surf.blit(scope_lbl, (cx, row_y))
            cx += scope_lbl.get_width() + 4
            scope_text_surf = small_font.render(str(scope_team), True, scope_color)
            panel_surf.blit(scope_text_surf, (cx, row_y))

            row_y += row_h

        draw_team_row(0)
        draw_team_row(1)

        surface.blit(panel_surf, panel_rect)

    def draw_message_log(self, surface):
        """Draw message log with scrolling support"""
        # Draw background
        pygame.draw.rect(surface, DARK_BLUE, self.message_log_rect, border_radius=5)

        # Header area for dragging and controls
        header_rect = pygame.Rect(
            self.message_log_rect.left,
            self.message_log_rect.top,
            self.message_log_rect.width,
            self.message_header_height,
        )
        pygame.draw.rect(surface, DARK_BLUE, header_rect, border_radius=5)
        # Minimize button at right of header
        btn_size = self.message_header_height - 8
        self.message_minimize_rect = pygame.Rect(
            header_rect.right - btn_size - 6,
            header_rect.top + 4,
            btn_size,
            btn_size,
        )
        pygame.draw.rect(surface, GOLD, self.message_minimize_rect, border_radius=4)
        minus_y = self.message_minimize_rect.centery
        pygame.draw.line(surface, DARK_BLUE, (self.message_minimize_rect.left + 4, minus_y), (self.message_minimize_rect.right - 4, minus_y), 3)

        # Draw border LAST, so it stays on top (fix top light blue outline)
        pygame.draw.rect(surface, LIGHT_BLUE, self.message_log_rect, 2, border_radius=5)

        # Resize handle at bottom-right
        self.message_resize_rect = pygame.Rect(
            self.message_log_rect.right - self.message_resize_zone_size,
            self.message_log_rect.bottom - self.message_resize_zone_size,
            self.message_resize_zone_size,
            self.message_resize_zone_size,
        )
        pygame.draw.rect(surface, LIGHT_GRAY, self.message_resize_rect, border_radius=3)
        
        # Draw title
        title_surf = self.small_font.render("Messages", True, WHITE)
        title_rect = title_surf.get_rect(midleft=(self.message_log_rect.left + 8, self.message_log_rect.top + 6))
        surface.blit(title_surf, title_rect)

        if self.message_minimized:
            return
        
        # Calcola area utile per i messaggi
        content_rect = pygame.Rect(
            self.message_log_rect.left + 5,
            self.message_log_rect.top + self.message_header_height + 5,
            self.message_log_rect.width - 30,  # Spazio per la scrollbar
            self.message_log_rect.height - self.message_header_height - 10
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
            scrollbar_outer = pygame.Rect(
                self.message_log_rect.right - 20,
                content_rect.top,
                15,
                content_rect.height
            )
            pygame.draw.rect(surface, LIGHT_GRAY, scrollbar_outer, border_radius=7)

            # Frecce di scorrimento dentro il box messaggi
            arrow_h = 16
            up_arrow = pygame.Rect(scrollbar_outer.left, content_rect.top, scrollbar_outer.width, arrow_h)
            down_arrow = pygame.Rect(scrollbar_outer.left, content_rect.bottom - arrow_h, scrollbar_outer.width, arrow_h)
            pygame.draw.rect(surface, LIGHT_GRAY, up_arrow, border_radius=3)
            pygame.draw.rect(surface, LIGHT_GRAY, down_arrow, border_radius=3)
            
            # Triangoli per le frecce
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

            # Traccia della scrollbar tra le frecce
            track_top = up_arrow.bottom + 2
            track_height = max(10, content_rect.height - 2 * arrow_h - 4)
            scrollbar_track = pygame.Rect(scrollbar_outer.left, track_top, scrollbar_outer.width, track_height)
            pygame.draw.rect(surface, (200, 200, 200), scrollbar_track, border_radius=7)

            # Cursore proporzionale alla quantità visibile
            if max_offset > 0:
                thumb_height = max(30, track_height * max_visible / total_messages)
                thumb_offset = (track_height - thumb_height) * (self.message_scroll_offset / max_offset)
            else:
                thumb_height = track_height
                thumb_offset = 0
            thumb_rect = pygame.Rect(scrollbar_track.left, scrollbar_track.top + thumb_offset, scrollbar_track.width, thumb_height)
            pygame.draw.rect(surface, WHITE, thumb_rect, border_radius=7)

            # Salva i rettangoli per click/drag
            # Persist scrollbar geometry for interactions
            self.scroll_up_rect = up_arrow
            self.scroll_down_rect = down_arrow
            self.scrollbar_rect = scrollbar_track
            self.scrollbar_thumb_rect = thumb_rect
            self.scrollbar_thumb_height = thumb_height
            self.scrollbar_max_offset = max_offset
        else:
            self.scroll_up_rect = None
            self.scroll_down_rect = None
            self.scrollbar_rect = None
            self.scrollbar_thumb_rect = None
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
        # In modalità a punti mostra anche i totali cumulativi di serie
        if self.series_mode == "points" and self.points_history:
            team0_score = sum(p0 for p0, _ in self.points_history)
            team1_score = sum(p1 for _, p1 in self.points_history)
        else:
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

        # Final series recap (points or hands)
        recap_y = winner_rect.bottom + int(height * 0.015)
        if self.series_mode == "points":
            final_text = f"Punteggio finale: {team0_score} - {team1_score}"
            if getattr(self, "series_target_points", None):
                final_text += f" (target {self.series_target_points})"
            final_surf = category_font.render(final_text, True, WHITE)
            final_rect = final_surf.get_rect(midtop=(panel_rect.centerx, recap_y))
            surface.blit(final_surf, final_rect)
            recap_y = final_rect.bottom + int(height * 0.01)
        elif self.series_mode == "hands":
            won0 = self.hands_won[0] if hasattr(self, "hands_won") else 0
            won1 = self.hands_won[1] if hasattr(self, "hands_won") else 0
            total_hands = self.series_num_hands if hasattr(self, "series_num_hands") else 0
            final_text = f"Mani vinte: {won0} - {won1} / {total_hands}"
            final_surf = category_font.render(final_text, True, WHITE)
            final_rect = final_surf.get_rect(midtop=(panel_rect.centerx, recap_y))
            surface.blit(final_surf, final_rect)
            recap_y = final_rect.bottom + int(height * 0.01)
        
        # Draw score breakdown with responsive positioning
        # Team 0 column (left side)
        team0_title = category_font.render("Team 0 Score", True, LIGHT_BLUE)
        team0_title_rect = team0_title.get_rect(
            topleft=(panel_rect.left + panel_width * 0.05, panel_rect.top + panel_height * 0.25)
        )
        surface.blit(team0_title, team0_title_rect)
        
        score_x = panel_rect.left + width * 0.07
        score_y = team0_title_rect.bottom + height * 0.02
        
        rules = self.app.game_config.get("rules", {})
        for category, score in self.final_breakdown[0].items():
            if category == "total":
                continue
            if category == "napola" and not rules.get("napola", False):
                continue
            if category == "re_bello" and not rules.get("re_bello", False):
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
            if category == "napola" and not rules.get("napola", False):
                continue
            if category == "re_bello" and not rules.get("re_bello", False):
                continue
            text = f"{category.capitalize()}: {score}"
            text_surf = detail_font.render(text, True, WHITE)
            surface.blit(text_surf, (score_x, score_y))
            score_y += height * 0.04
        
        # Total for team 1
        total1_text = f"Total: {self.final_breakdown[1]['total']}"
        total1_surf = category_font.render(total1_text, True, HIGHLIGHT_RED)
        surface.blit(total1_surf, (score_x, score_y + height * 0.02))
        
        # Draw exit button (always Exit)
        button_width = width * 0.2
        button_height = height * 0.07
        new_game_button = Button(
            panel_rect.centerx - button_width // 2,
            panel_rect.bottom - button_height - height * 0.03,
            button_width, button_height,
            "Exit",
            DARK_GREEN, WHITE,
            font_size=int(height * 0.03)
        )
        new_game_button.draw(surface)
        
        # Store button rect for click detection
        self.game_over_button_rect = new_game_button.rect
        
    def is_current_player_controllable(self):
        """Verifica robusta per il controllo del giocatore corrente"""
        mode = self.app.game_config.get("mode")
        
        # Log per debugging
        #print(f"### VERIFICA CONTROLLO GIOCATORE ###")
        #print(f"Giocatore corrente: {self.current_player_id}, Locale: {self.local_player_id}")
        #print(f"Modalità: {mode}, Tipo online: {self.app.game_config.get('online_type')}")
        
        # Caso base: sempre controllabile se è il proprio turno
        if self.current_player_id == self.local_player_id:
            #print("È il turno del giocatore locale")
            return True
        
        # Altre modalità di gioco
        if mode == "online_multiplayer":
            # Online: mai controllare giocatori diversi da sé stessi
            return False
        elif mode == "team_vs_ai":
            # Locale team vs AI: controllo dei giocatori 0 e 2
            return self.current_player_id in [0, 2]
        elif mode == "local_multiplayer":
            # Modalità locale: tutti controllabili
            return True
        else:
            # Altre modalità: solo giocatore locale
            return self.current_player_id == self.local_player_id

    def draw_intermediate_recap(self, surface):
        """Overlay di recap dell'ultima mano in modalità a punti con pulsante Avanti."""
        width = self.app.window_width
        height = self.app.window_height
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        panel_w = int(width * 0.6)
        panel_h = int(height * 0.5)
        panel_rect = pygame.Rect(width//2 - panel_w//2, height//2 - panel_h//2, panel_w, panel_h)
        pygame.draw.rect(surface, DARK_BLUE, panel_rect, border_radius=10)
        pygame.draw.rect(surface, GOLD, panel_rect, 4, border_radius=10)

        title_font = pygame.font.SysFont(None, int(height * 0.05))
        detail_font = pygame.font.SysFont(None, int(height * 0.03))

        title = title_font.render("Fine mano - riepilogo", True, GOLD)
        surface.blit(title, title.get_rect(midtop=(panel_rect.centerx, panel_rect.top + int(height * 0.03))))

        bd = self.last_hand_breakdown or {}
        p0 = int(bd.get(0, {}).get("total", 0))
        p1 = int(bd.get(1, {}).get("total", 0))
        tot0 = sum(x for x, _ in self.points_history)
        tot1 = sum(x for _, x in self.points_history)

        # Riga punteggi mano
        hand_text = detail_font.render(f"Mano: {p0} - {p1}", True, WHITE)
        surface.blit(hand_text, hand_text.get_rect(midtop=(panel_rect.centerx, panel_rect.top + int(height * 0.14))))

        # Riga cumulativi
        if self.series_mode == "points":
            extra = f" (target {self.series_target_points})"
        elif self.series_mode == "hands":
            extra = f" | Mani vinte: {self.hands_won[0]}-{self.hands_won[1]} / {self.series_num_hands}"
        else:
            extra = ""
        tot_text = detail_font.render(f"Totale serie: {tot0} - {tot1}{extra}", True, WHITE)
        surface.blit(tot_text, tot_text.get_rect(midtop=(panel_rect.centerx, panel_rect.top + int(height * 0.2))))

        # Pulsanti Avanti ed Exit sull'overlay
        btn_w = int(panel_w * 0.28)
        btn_h = int(height * 0.06)
        gap = int(width * 0.02)
        left_x = panel_rect.centerx - btn_w - gap//2
        right_x = panel_rect.centerx + gap//2
        y_btn = panel_rect.bottom - btn_h - int(height * 0.04)
        self.next_hand_button = Button(left_x, y_btn, btn_w, btn_h, "Avanti", DARK_GREEN, WHITE, font_size=int(height * 0.03))
        self.exit_overlay_button = Button(right_x, y_btn, btn_w, btn_h, "Exit", HIGHLIGHT_RED, WHITE, font_size=int(height * 0.03))
        self.next_hand_button.draw(surface)
        self.exit_overlay_button.draw(surface)

    # Helper: host broadcast of series state
    def _broadcast_series_state(self):
        if not (self.app.game_config.get("mode") == "online_multiplayer" and self.app.game_config.get("is_host")):
            return
        payload = {
            'game_state': self.env.game_state.copy() if self.env and hasattr(self.env, 'game_state') else {},
            'current_player': self.env.current_player if self.env and hasattr(self.env, 'current_player') else None,
            'series_mode': self.series_mode,
            'series_target_points': self.series_target_points,
            'series_num_hands': self.series_num_hands,
            'series_tiebreak': self.series_tiebreak,
            'series_scores': self.series_scores,
            'series_hands_played': self.series_hands_played,
            'points_history': self.points_history,
            'hands_won': self.hands_won,
            'show_intermediate_recap': self.show_intermediate_recap,
            'last_hand_breakdown': self.last_hand_breakdown,
            '_pending_next_starter': self._pending_next_starter,
        }
        if hasattr(self.app, 'network') and self.app.network:
            try:
                self.app.network.broadcast_series_state(payload)
            except Exception:
                pass
    
    def start_replay(self):
        """Start replay of the last 3 moves from opponents (or fewer if game just started)"""
        if not self.env or self.replay_active:
            return
        
        # Get the last moves from game history
        history = self.env.game_state.get("history", [])
        if not history:
            return  # No moves to replay
        
        # Filter moves to only include opponent moves (not the user's moves)
        mode = self.app.game_config.get("mode")
        # Determina l'ID locale in modo robusto (specialmente per client online)
        effective_local_id = self.local_player_id
        if mode == "online_multiplayer" and hasattr(self.app, "network") and self.app.network:
            net_player_id = getattr(self.app.network, "player_id", None)
            if net_player_id is not None:
                effective_local_id = net_player_id
        opponent_moves = []
        
        for move in history:
            player = move["player"]
            is_opponent = False
            
            if mode == "single_player":
                # In single player, opponents are players 1, 2, 3
                is_opponent = player != 0
            elif mode == "team_vs_ai":
                # In team vs AI locale, escludi solo le mosse del giocatore attualmente controllato
                # (così vediamo anche quelle del partner umano quando stiamo guardando come partner)
                is_opponent = player != self.current_player_id
            elif mode == "local_multiplayer":
                # In local multiplayer, all other players are opponents
                is_opponent = player != self.current_player_id
            elif mode == "online_multiplayer":
                # In online multiplayer, all other players are opponents rispetto al locale
                is_opponent = player != effective_local_id
            
            if is_opponent:
                opponent_moves.append(move)
        
        # Get the last 3 opponent moves (or fewer if there are less)
        num_moves = min(3, len(opponent_moves))
        self.replay_moves = opponent_moves[-num_moves:]
        
        if not self.replay_moves:
            return
        
        # Calculate the table state at the beginning of the replay
        # We need to find the state before the first replay move
        first_replay_move_index = history.index(self.replay_moves[0])
        self.replay_table_state = []
        
        # Reconstruct the table state by going through all moves before the first replay move
        # Start with initial table state (empty)
        table_state = []
        
        for move in history[:first_replay_move_index]:
            captured = move.get("captured_cards") or []
            if captured:
                # On a capture, the played card does NOT stay on table; only remove captured table cards
                for card in captured:
                    if card in table_state:
                        table_state.remove(card)
            else:
                # No capture: the played card remains on table
                table_state.append(move["played_card"])
        
        self.replay_table_state = table_state
        
        # Start replay
        self.replay_active = True
        self.replay_current_index = 0
        
        # NUOVO: Inizializza il set di carte in movimento per il replay
        if not hasattr(self, 'replay_cards_in_motion'):
            self.replay_cards_in_motion = set()
        else:
            self.replay_cards_in_motion.clear()
        # NUOVO: Pulisci anche eventuali carte in movimento del gioco normale
        if hasattr(self, 'cards_in_motion'):
            self.cards_in_motion.clear()
        
        # Inizializza una breve pausa per mostrare lo stato iniziale del tavolo
        self.replay_initial_pause = 20  # ~1/3 di secondo a 60 FPS
        # Il primo movimento del replay verrà avviato da update_replay_animations dopo la pausa
    
    def play_next_replay_move(self):
        """Play the next move in the replay sequence"""
        if not self.replay_active or self.replay_current_index >= len(self.replay_moves):
            self.end_replay()
            return
        
        move = self.replay_moves[self.replay_current_index]
        
        # Non modifichiamo più subito lo stato del tavolo del replay.
        # Le modifiche al tavolo avverranno sincronizzate con la fine delle animazioni
        # (aggiunta dopo mano->tavolo, rimozione dopo cattura).
        
        # Create animations for this move
        # Nota: lo stato logico del tavolo (replay_table_state) viene aggiornato
        # in sincronizzazione con la fine delle animazioni (no-capture: aggiunta carta giocata
        # alla fine della play; capture: rimozione carte catturate alla fine delle animazioni di cattura)
        self.create_replay_animations(move)
        
        self.replay_current_index += 1
    
    def create_replay_animations(self, move):
        """Create animations for a replay move using the same parameters as regular game animations"""
        player = move["player"]
        played_card = move["played_card"]
        captured_cards = move["captured_cards"]
        
        # Get player's visual position
        visual_pos = self.get_visual_position(player)
        
        # Calculate start position (from player's hand) - same as regular game
        if visual_pos == 0:  # Bottom player
            start_x = self.app.window_width // 2
            start_y = self.app.window_height - self.card_height - self.app.window_height * 0.05
        elif visual_pos == 1:  # Left player
            start_x = self.app.window_width * 0.02
            start_y = self.app.window_height // 2
        elif visual_pos == 2:  # Top player
            start_x = self.app.window_width // 2
            start_y = self.app.window_height * 0.05
        else:  # Right player
            start_x = self.app.window_width - self.card_width - self.app.window_width * 0.02
            start_y = self.app.window_height // 2
        
        start_pos = (start_x, start_y)
        
        # Calculate end position (center of table)
        table_center = (self.app.window_width // 2, self.app.window_height // 2)
        end_pos = table_center  # Default: center of table
        
        # If there are captured cards, calculate position like in regular game
        if captured_cards:
            # Calculate table positions like in regular game
            width = self.app.window_width
            card_width = int(width * 0.078)
            card_height = int(card_width * 1.5)
            
            # Use the replay table state to find card positions
            original_table = self.replay_table_state.copy()
            for card in captured_cards:
                if card not in original_table:
                    original_table.append(card)
            
            max_spacing = self.table_rect.width * 0.8 / max(len(original_table), 1)
            card_spacing = min(card_width * 1.1, max_spacing)
            start_x = self.table_rect.centerx - (len(original_table) * card_spacing) // 2
            table_y = self.table_rect.centery - card_height // 2
            
            # Find the leftmost captured card
            leftmost_card_index = float('inf')
            for card in captured_cards:
                try:
                    idx = original_table.index(card)
                    if idx < leftmost_card_index:
                        leftmost_card_index = idx
                except ValueError:
                    pass
            
            if leftmost_card_index != float('inf'):
                # Position the played card to partially overlap the leftmost captured card
                card_x = start_x + leftmost_card_index * card_spacing
                end_pos = (card_x - card_width * 0.25, table_y + card_height / 2)
        
        # Use the same animation parameters as regular game animations
        hand_to_table_duration = 15   # Same as regular game
        plateau_duration = 30         # Same as regular game
        capture_duration = 25         # Same as regular game
        inter_card_delay = 10         # Same as regular game
        
        # Phase 1: Card from hand to table
        hand_to_table = CardAnimation(
            card=played_card,
            start_pos=start_pos,
            end_pos=end_pos,
            duration=hand_to_table_duration,
            delay=0,
            scale_start=1.0,
            scale_end=1.0,
            rotation_start=0,
            rotation_end=0,
            animation_type="replay_play"
        )
        # Indica se questa giocata causerà una presa
        try:
            hand_to_table.causes_capture = bool(captured_cards)
        except Exception:
            hand_to_table.causes_capture = False
        self.replay_animations.append(hand_to_table)
        
        # If there are captured cards, create capture animations
        if captured_cards:
            # Calculate pile rect for the capturing team (use new pile positions)
            current_player = self.players[player]
            team_id = current_player.team_id
            pile_rect = self.get_team_pile_rect(team_id)
            
            # Phase 2: Plateau animation (keep card visible)
            plateau_anim = CardAnimation(
                card=played_card,
                start_pos=end_pos,
                end_pos=end_pos,
                duration=plateau_duration,
                delay=hand_to_table_duration,
                scale_start=1.0,
                scale_end=1.0,
                rotation_start=0,
                rotation_end=0,
                animation_type="replay_plateau"
            )
            self.replay_animations.append(plateau_anim)
            
            # Phase 3: Capture animations
            total_time = hand_to_table_duration + plateau_duration
            all_capture_cards = [played_card] + list(captured_cards)
            
            # Calculate starting positions for all cards
            starting_positions = {}
            starting_positions[played_card] = end_pos
            
            # Calculate positions of cards on the table
            for card in captured_cards:
                try:
                    card_index = original_table.index(card)
                    card_x = start_x + card_index * card_spacing
                    card_pos = (card_x + card_width // 2, table_y + card_height // 2)
                    starting_positions[card] = card_pos
                except ValueError:
                    starting_positions[card] = table_center
            
            # Create capture animations for each card
            for i, card in enumerate(all_capture_cards):
                # Slight variation in final position to avoid overlap
                staggered_offset = card_width * 0.3
                end_x = pile_rect.centerx + random.randint(-5, 5) + i * staggered_offset
                end_y = pile_rect.centery + random.randint(-5, 5) + i * 3
                varied_end_pos = (end_x, end_y)
                
                # Calculate delay for this card
                card_delay = total_time + i * inter_card_delay
                
                # NUOVO: Crea un'animazione speciale "start_motion" per tracciare l'inizio del movimento
                motion_start_anim = CardAnimation(
                    card=card,
                    start_pos=starting_positions[card],  # Stessa posizione di partenza
                    end_pos=starting_positions[card],    # Stessa posizione (non si muove)
                    duration=1,                          # Dura solo 1 frame
                    delay=card_delay,                    # Stesso delay dell'animazione di cattura
                    scale_start=1.0,
                    scale_end=1.0,
                    rotation_start=0,
                    rotation_end=0,
                    animation_type="replay_start_motion"  # Tipo speciale per tracciare l'inizio del movimento
                )
                self.replay_animations.append(motion_start_anim)
                
                capture_anim = CardAnimation(
                    card=card,
                    start_pos=starting_positions[card],
                    end_pos=varied_end_pos,
                    duration=capture_duration,
                    delay=card_delay,
                    scale_start=1.0,
                    scale_end=0.8,
                    rotation_start=0,
                    rotation_end=random.randint(-10, 10),
                    animation_type="replay_capture"
                )
                self.replay_animations.append(capture_anim)
    
    def update_replay_animations(self):
        """Update replay animations and trigger next move when current animations complete"""
        if not self.replay_active:
            return
        
        # NUOVO: Pausa iniziale per mostrare lo stato del tavolo prima delle mosse
        if hasattr(self, 'replay_initial_pause') and self.replay_initial_pause > 0:
            self.replay_initial_pause -= 1
            if self.replay_initial_pause == 0:
                # Avvia il primo movimento del replay
                self.play_next_replay_move()
            return
        
        # NUOVO: Assicurati che esista il set di carte in movimento per il replay
        if not hasattr(self, 'replay_cards_in_motion'):
            self.replay_cards_in_motion = set()
        
        # NUOVO: Lista temporanea per le carte da rimuovere dal set di movimento del replay
        replay_cards_to_remove = []
        
        # Update all replay animations
        completed_animations = []
        for anim in self.replay_animations:
            # Verifica se questa è un'animazione di inizio movimento
            if anim.animation_type == "replay_start_motion" and anim.current_frame == 0:
                # Aggiungi la carta al set delle carte in movimento del replay
                self.replay_cards_in_motion.add(anim.card)
                #print(f"Replay: Carta {anim.card} aggiunta al set di carte in movimento")
            
            # Quando termina l'animazione mano->tavolo, aggiungi la carta al tavolo del replay SOLO per mosse senza cattura
            if anim.animation_type == "replay_play" and anim.current_frame == anim.duration - 1:
                causes_capture = getattr(anim, 'causes_capture', False)
                if not causes_capture and anim.card not in self.replay_table_state:
                    self.replay_table_state.append(anim.card)

            # Verifica se questa è un'animazione di cattura che termina
            if anim.animation_type == "replay_capture" and anim.current_frame == anim.duration - 1:
                # La carta catturata non deve restare sul tavolo di replay
                if anim.card in self.replay_table_state:
                    try:
                        self.replay_table_state.remove(anim.card)
                    except ValueError:
                        pass
                # Aggiungi la carta alla lista di rimozione invece di rimuoverla immediatamente dal set motion
                replay_cards_to_remove.append(anim.card)
                #print(f"Replay: Carta {anim.card} marcata per rimozione dal set di carte in movimento")
            # Non serve rimuovere la carta giocata per mosse con cattura qui: non viene mai aggiunta
            
            if anim.update():
                completed_animations.append(anim)
        
        # Remove completed animations
        for anim in completed_animations:
            self.replay_animations.remove(anim)
        
        # NUOVO: Rimuovi le carte dal set di movimento del replay solo se non ci sono più animazioni attive
        # Questo evita il flash delle carte che riappaiono sul tavolo durante il replay
        if not self.replay_animations and replay_cards_to_remove:
            for card in replay_cards_to_remove:
                if card in self.replay_cards_in_motion:
                    self.replay_cards_in_motion.remove(card)
                    #print(f"Replay: Carta {card} rimossa dal set di carte in movimento (dopo completamento animazioni)")
        
        # If all animations are done, play next move or end replay
        if not self.replay_animations:
            if self.replay_current_index < len(self.replay_moves):
                self.play_next_replay_move()
            else:
                self.end_replay()
    
    def end_replay(self):
        """End replay"""
        self.replay_active = False
        self.replay_moves = []
        self.replay_current_index = 0
        self.replay_animations = []
        self.replay_table_state = []
        # NUOVO: Pulisci anche il set di carte in movimento del replay
        if hasattr(self, 'replay_cards_in_motion'):
            self.replay_cards_in_motion.clear()
    
    def draw_replay_animations(self, surface):
        """Draw replay animations with rounded corners and borders"""
        # Border parameters
        border_thickness = 1  # Sottile bordo nero
        border_radius = 8     # Angoli smussati
        
        for anim in self.replay_animations:
            if anim.current_frame < 0:  # Still in delay
                continue
                
            current_pos = anim.get_current_pos()
            current_scale = anim.get_current_scale()
            current_rotation = anim.get_current_rotation()
            
            # Get card image
            card_img = self.app.resources.get_card_image(anim.card)
            if card_img:
                # Scale the image
                scaled_width = int(self.card_width * current_scale)
                scaled_height = int(self.card_height * current_scale)
                scaled_img = pygame.transform.scale(card_img, (scaled_width, scaled_height))
                
                # Compose rounded card with border BEFORE rotation
                base_card = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
                base_card.fill((0, 0, 0, 0))
                base_rect = base_card.get_rect()
                pygame.draw.rect(base_card, (255, 255, 255), base_rect, border_radius=border_radius)
                base_card.blit(scaled_img, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                pygame.draw.rect(base_card, BLACK, base_rect, width=2, border_radius=border_radius)
                
                # Apply rotation
                rotated_img = pygame.transform.rotate(base_card, current_rotation)
                
                # Create rect centered at position and blit rotated card directly
                rect = rotated_img.get_rect(center=current_pos)
                surface.blit(rotated_img, rect)

class ScoponeApp:
    """Main application class"""
    def __init__(self):
        pygame.init()
        # Initialize clipboard support (pygame.scrap) if available
        try:
            if hasattr(pygame, 'scrap'):
                pygame.scrap.init()
        except Exception:
            pass
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
            "options": GameOptionsScreen(self),
            "lobby": None,  # Placeholder, initialized lazily to avoid font init clashes
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
            # Lazy-init LobbyScreen when needed
            if screen is None and self.current_screen == "lobby":
                screen = LobbyScreen(self)
                self.screens["lobby"] = screen
            
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
        #print(f"Resized window to: {self.window_width}x{self.window_height}")
        
        # Scale background image from the original
        if hasattr(self.resources, 'original_background') and self.resources.original_background:
            #print("Scaling background from original image")
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