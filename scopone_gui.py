import pygame
import sys
import os
import itertools

from environment import ScoponeEnvMA
from actions import encode_action, get_valid_actions, MAX_ACTIONS

# Dizionari per i nomi completi di rank e suit
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
SUIT_NAMES = {
    'denari': 'denari',
    'coppe': 'coppe',
    'spade': 'spade',
    'bastoni': 'bastoni'
}

def format_card(card):
    """
    Converte una tupla (rank, suit) nel nome base del file immagine.
    Esempio: (1, 'denari') -> "01_Asso_di_denari"
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

def load_card_images(folder="assets"):
    """
    Carica e restituisce un dizionario che mappa (rank, suit) -> immagine Pygame
    basato sul nome file generato da format_card, con estensione .jpg.
    """
    images_dict = {}
    possible_ranks = [1,2,3,4,5,6,7,8,9,10]
    possible_suits = ['denari','coppe','spade','bastoni']
    for r in possible_ranks:
        for s in possible_suits:
            key = (r,s)
            filename_key = format_card(key)  # ad es. "01_Asso_di_denari"
            filename = os.path.join(folder, filename_key + ".jpg")
            if os.path.isfile(filename):
                img = pygame.image.load(filename)
                # Ridimensiona le immagini delle carte a CARD_W x CARD_H
                card_img = pygame.transform.scale(img, (60, 90))
                images_dict[key] = card_img
            else:
                # Se manca il file, puoi usare un placeholder (qui disegniamo un rettangolo)
                pass
    return images_dict

def format_move(move):
    """
    Restituisce una stringa testuale per debug, ad es.:
    "Player 2 => played (01_Asso_di_denari), capture=scopa, captured=[04_Quattro_di_denari 17_Sette_di_coppe]"
    """
    pl = move["player"]
    played_str = format_card(move["played_card"])
    ctype = move["capture_type"]
    captured_str = " ".join(format_card(c) for c in move.get("captured_cards", []))
    return f"Player {pl} => played ({played_str}), capture={ctype}, captured=[{captured_str}]"

# Impostazioni della finestra e grafica
WIN_WIDTH = 1000
WIN_HEIGHT = 700   # Aumentata l'altezza per il riepilogo finale
FPS = 30

# Colori
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY  = (200,200,200)
RED   = (255,0,0)
GREEN = (0,200,0)
BLUE  = (80,80,255)

# Dimensioni carta
CARD_W = 60
CARD_H = 90

# Area del riepilogo finale
SUMMARY_AREA_Y = 480  # Y a partire da cui viene disegnata la summary

class Button:
    """
    Bottone testuale semplificato.
    """
    def __init__(self, x, y, w, h, text, color_bg=BLUE, color_fg=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color_bg = color_bg
        self.color_fg = color_fg
        self.font = pygame.font.SysFont(None, 20)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color_bg, self.rect)
        render = self.font.render(self.text, True, self.color_fg)
        surface.blit(render, (self.rect.x+5, self.rect.y+5))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# --- Funzioni per disegnare testo e interfaccia ---

def draw_text(surface, text, x, y, color=BLACK, font=None):
    if font is None:
        font = pygame.font.SysFont(None, 20)
    render = font.render(text, True, color)
    surface.blit(render, (x, y))

def draw_final_summary(surface, card_images, env, final_breakdown):
    """
    Disegna la schermata finale con:
      - Le immagini delle carte catturate da Team 0 e Team 1.
      - Il breakdown dei punti con spiegazione.
    """
    surface.fill((240,240,240))
    title_font = pygame.font.SysFont(None, 32)
    info_font = pygame.font.SysFont(None, 24)

    draw_text(surface, "=== RISULTATI FINALI ===", 50, 20, RED, title_font)
    
    gs = env.game_state
    captured = gs["captured_squads"]

    # Disegna le carte catturate per Team 0
    draw_text(surface, "Squadra 0 ha catturato:", 50, 70, BLUE, info_font)
    x0 = 50
    y0 = 110
    gap = 70
    for i, card in enumerate(captured[0]):
        rx = x0 + (i % 10) * gap
        ry = y0 + (i // 10) * (CARD_H + 10)
        if card in card_images:
            surface.blit(card_images[card], (rx, ry))
        else:
            pygame.draw.rect(surface, GRAY, (rx, ry, CARD_W, CARD_H))
            draw_text(surface, format_card(card), rx+5, ry+5, BLACK, info_font)

    # Disegna le carte catturate per Team 1
    draw_text(surface, "Squadra 1 ha catturato:", 50, 300, BLUE, info_font)
    x1 = 50
    y1 = 340
    for i, card in enumerate(captured[1]):
        rx = x1 + (i % 10) * gap
        ry = y1 + (i // 10) * (CARD_H + 10)
        if card in card_images:
            surface.blit(card_images[card], (rx, ry))
        else:
            pygame.draw.rect(surface, GRAY, (rx, ry, CARD_W, CARD_H))
            draw_text(surface, format_card(card), rx+5, ry+5, BLACK, info_font)

    # Disegna il breakdown dei punti
    if final_breakdown:
        b0 = final_breakdown[0]
        b1 = final_breakdown[1]
        summary_y = 520
        text0 = (f"Team 0 => Carte: {b0['carte']}, Denari: {b0['denari']}, "
                 f"Settebello: {b0['settebello']}, Primiera: {b0['primiera']}, "
                 f"Scope: {b0['scope']} => Totale: {b0['total']}")
        text1 = (f"Team 1 => Carte: {b1['carte']}, Denari: {b1['denari']}, "
                 f"Settebello: {b1['settebello']}, Primiera: {b1['primiera']}, "
                 f"Scope: {b1['scope']} => Totale: {b1['total']}")
        draw_text(surface, text0, 50, summary_y, BLACK, info_font)
        draw_text(surface, text1, 50, summary_y + 30, BLACK, info_font)
    
    pygame.display.update()

# --- Main GUI con Pygame ---

def main_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Scopone a Coppie - Pygame - Carte Immagini")
    clock = pygame.time.Clock()

    # Carica le immagini delle carte
    card_images = load_card_images(folder="assets")

    env = None
    done = False
    show_final_summary = False
    final_breakdown = None
    msg_text = "Premi 'Nuova Partita' per iniziare."

    selected_hand_index = None
    selected_table_indices = set()

    btn_new_game = Button(20, 20, 120, 30, "Nuova Partita", BLUE, WHITE)
    btn_confirm  = Button(20, 60, 120, 30, "Conferma Mossa", GREEN, WHITE)

    font_msg = pygame.font.SysFont(None, 22)
    font_info = pygame.font.SysFont(None, 18)

    def reset_env():
        nonlocal env, done, show_final_summary, final_breakdown, msg_text
        nonlocal selected_hand_index, selected_table_indices
        env = ScoponeEnvMA()
        env.reset()
        done = False
        show_final_summary = False
        final_breakdown = None
        msg_text = "Nuova partita iniziata."
        selected_hand_index = None
        selected_table_indices.clear()

    def draw_text(surface, text, x, y, color=(0,0,0), font=font_msg):
        render = font.render(text, True, color)
        surface.blit(render, (x, y))

    def draw_scene():
        screen.fill(WHITE)
        btn_new_game.draw(screen)
        btn_confirm.draw(screen)
        draw_text(screen, msg_text, 160, 30, RED)

        if env is not None:
            cp = env.current_player
            gs = env.game_state

            # Info a sinistra
            lines = []
            lines.append(f"Giocatore di turno: {cp}")
            for p in range(4):
                lines.append(f"  Giocatore {p} => {len(gs['hands'][p])} carte")
            lines.append(f" Tavolo => {len(gs['table'])} carte")
            if gs["history"]:
                lines.append("Ultime mosse:")
                for move in gs["history"][-2:]:
                    lines.append("  " + format_move(move))
            oy = 110
            for ln in lines:
                draw_text(screen, ln, 20, oy, BLACK, font_info)
                oy += 22

            # Disegna le carte in mano del giocatore corrente
            if not done:
                hand = gs["hands"][cp]
                base_x = 150
                base_y = 450
                gap = 70
                for i, c in enumerate(hand):
                    x_i = base_x + i * gap
                    y_i = base_y
                    if c in card_images:
                        screen.blit(card_images[c], (x_i, y_i))
                        if i == selected_hand_index:
                            pygame.draw.rect(screen, GREEN, (x_i, y_i, CARD_W, CARD_H), 3)
                    else:
                        pygame.draw.rect(screen, GRAY, (x_i, y_i, CARD_W, CARD_H))
                        draw_text(screen, format_card(c), x_i+5, y_i+5, BLACK, font_info)

                # Disegna le carte sul tavolo
                table_cards = gs["table"]
                base_tx = 220
                base_ty = 150
                gap_t = 70
                for i, c in enumerate(table_cards):
                    x_i = base_tx + i * gap_t
                    y_i = base_ty
                    if c in card_images:
                        screen.blit(card_images[c], (x_i, y_i))
                        if i in selected_table_indices:
                            pygame.draw.rect(screen, RED, (x_i, y_i, CARD_W, CARD_H), 3)
                    else:
                        pygame.draw.rect(screen, GRAY, (x_i, y_i, CARD_W, CARD_H))
                        draw_text(screen, format_card(c), x_i+5, y_i+5, BLACK, font_info)

        pygame.display.update()

    running = True
    while running:
        clock.tick(FPS)
        if show_final_summary:
            draw_final_summary(screen, card_images, env, final_breakdown)
        else:
            draw_scene()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if btn_new_game.is_clicked(pos):
                    reset_env()
                elif btn_confirm.is_clicked(pos):
                    if env is None or done:
                        msg_text = "Partita non avviata o già terminata."
                    else:
                        if selected_hand_index is None:
                            msg_text = "Seleziona una carta in mano prima di confermare."
                        else:
                            valid_acts = env.get_valid_actions()
                            act_id = encode_action(selected_hand_index, sorted(selected_table_indices))
                            if act_id not in valid_acts:
                                msg_text = "Mossa NON valida. Riprova."
                            else:
                                obs_next, reward, done_flag, info = env.step(act_id)
                                done = done_flag
                                if done:
                                    # Quando la partita finisce, prendiamo il breakdown dettagliato
                                    final_breakdown = info.get("score_breakdown", None)
                                    show_final_summary = True
                                    msg_text = "Partita terminata!"
                                else:
                                    msg_text = "Mossa effettuata."
                                selected_hand_index = None
                                selected_table_indices.clear()

                else:
                    if env and not done:
                        cp = env.current_player
                        gs = env.game_state
                        # Clic sulle carte in mano
                        hand = gs["hands"][cp]
                        base_x = 150
                        base_y = 450
                        gap = 70
                        for i, c in enumerate(hand):
                            rx = base_x + i * gap
                            ry = base_y
                            r_rect = pygame.Rect(rx, ry, CARD_W, CARD_H)
                            if r_rect.collidepoint(pos):
                                selected_hand_index = i
                                msg_text = f"Selezionata carta indice {i} in mano"
                                break
                        # Clic sulle carte sul tavolo
                        table_cards = gs["table"]
                        base_tx = 220
                        base_ty = 150
                        gap_t = 70
                        for i, c in enumerate(table_cards):
                            rx = base_tx + i * gap_t
                            ry = base_ty
                            r_rect = pygame.Rect(rx, ry, CARD_W, CARD_H)
                            if r_rect.collidepoint(pos):
                                if i in selected_table_indices:
                                    selected_table_indices.remove(i)
                                    msg_text = f"Rimossa carta tavolo indice {i}"
                                else:
                                    selected_table_indices.add(i)
                                    msg_text = f"Aggiunta carta tavolo indice {i}"
                                break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_pygame()
