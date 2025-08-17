import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # headless

import pygame

from scopone_gui import ResourceManager, SUIT_NAMES, RANK_NAMES, ensure_tuple

def test_gui_resource_manager_headless():
    pygame.init()
    try:
        res = ResourceManager()
        res.load_resources()
        # Probe a couple of card images using tuple and ID
        img1 = res.get_card_image((1, 'denari'))
        img2 = res.get_card_image(0)  # ID of (1,'denari')
        assert img1 is not None
        assert img2 is not None
    finally:
        pygame.quit()

