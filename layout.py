import pygame

class LayoutManager:
    """Gestisce il posizionamento degli elementi UI evitando sovrapposizioni"""
    
    @staticmethod
    def check_collision(rect1, rect2, min_distance=0):
        """
        Verifica se due rettangoli si sovrappongono o sono troppo vicini
        
        Args:
            rect1, rect2: I rettangoli da controllare
            min_distance: Distanza minima tra i rettangoli (in pixel)
        """
        # Espandi temporaneamente rect1 per includere la distanza minima
        expanded = rect1.inflate(min_distance*2, min_distance*2)
        return expanded.colliderect(rect2)
    
    @staticmethod
    def get_collision_free_position(rect, obstacles, fixed_zones=None, preferred_direction='down', padding=10):
        """
        Trova una posizione per rect che eviti sovrapposizioni con gli ostacoli.
        
        Args:
            rect: Il rettangolo da posizionare
            obstacles: Lista di rettangoli da evitare
            fixed_zones: Aree speciali da evitare completamente (es. tavolo di gioco)
            preferred_direction: Direzione preferita per lo spostamento ('up', 'down', 'left', 'right')
            padding: Spazio aggiuntivo da mantenere tra gli elementi
        
        Returns:
            Un nuovo pygame.Rect nella posizione senza collisioni
        """
        # Copia il rettangolo originale
        new_rect = rect.copy()
        
        # Se non ci sono ostacoli, ritorna la posizione originale
        if not obstacles:
            return new_rect
            
        # Calcola lo spazio totale necessario tra gli elementi
        min_distance = padding
        
        # Verifica se c'è già una collisione
        has_collision = any(LayoutManager.check_collision(new_rect, obs, min_distance) for obs in obstacles)
        if not has_collision:
            # Verifica anche le zone fisse se specificate
            if fixed_zones and any(LayoutManager.check_collision(new_rect, zone, min_distance*2) for zone in fixed_zones):
                has_collision = True
            else:
                return new_rect
        
        # Calcola lo spostamento necessario nella direzione preferita
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up-right': (1, -1),
            'up-left': (-1, -1),
            'down-right': (1, 1),
            'down-left': (-1, 1)
        }
        
        # Ordina le direzioni mettendo quella preferita per prima
        ordered_directions = [preferred_direction]
        for d in directions:
            if d != preferred_direction:
                ordered_directions.append(d)
                
        # Prova ciascuna direzione fino a trovare una posizione valida
        best_position = None
        min_distance_moved = float('inf')
        
        for direction in ordered_directions:
            dx, dy = directions[direction]
            test_rect = new_rect.copy()
            
            # Incrementa gradualmente lo spostamento finché non si trova una posizione valida
            for distance in range(min_distance, 500, 5):  # Parti dal padding minimo
                test_rect.x = new_rect.x + dx * distance
                test_rect.y = new_rect.y + dy * distance
                
                # Verifica collisioni con ostacoli e zone fisse
                if not any(LayoutManager.check_collision(test_rect, obs, min_distance) for obs in obstacles):
                    if fixed_zones and any(LayoutManager.check_collision(test_rect, zone, min_distance*2) for zone in fixed_zones):
                        continue
                        
                    # Calcola la distanza totale spostata
                    distance_moved = ((test_rect.x - new_rect.x)**2 + (test_rect.y - new_rect.y)**2)**0.5
                    
                    # Se questa è la prima posizione valida o è più vicina dell'ultima trovata
                    if distance_moved < min_distance_moved:
                        best_position = test_rect.copy()
                        min_distance_moved = distance_moved
                        
                        # Se è nella direzione preferita, accettiamo subito
                        if direction == preferred_direction:
                            return best_position
                            
                    break
        
        # Se abbiamo trovato una posizione valida, ritornala
        if best_position:
            return best_position
            
        # Altrimenti, ritorna la posizione originale (potrebbe ancora avere collisioni)
        return rect

    @staticmethod
    def arrange_elements(elements, screen_width, screen_height, fixed_zones=None, padding=10):
        """
        Dispone gli elementi evitando sovrapposizioni.
        
        Args:
            elements: Lista di tuple (rect, priority) dove:
                      - rect è il pygame.Rect dell'elemento
                      - priority è un intero che indica la priorità (più basso = più importante)
            screen_width: Larghezza dello schermo
            screen_height: Altezza dello schermo
            fixed_zones: Aree speciali da evitare completamente
            padding: Spazio tra gli elementi
        
        Returns:
            Lista di pygame.Rect riposizionati
        """
        if not elements:
            return []
            
        # Ordina gli elementi per priorità
        sorted_elements = sorted(elements, key=lambda x: x[1])
        
        # Estrai solo i rettangoli, mantieni l'ordine di priorità
        sorted_rects = [elem[0] for elem in sorted_elements]
        
        # Lista dei rettangoli già posizionati (inizialmente vuota)
        positioned_rects = []
        
        # Lista dei nuovi rettangoli da restituire
        new_rects = []
        
        # Scegli la direzione preferita in base alla posizione dell'elemento
        def choose_direction(rect):
            cx, cy = rect.center
            middle_x, middle_y = screen_width//2, screen_height//2
            
            # Scegli la direzione opposta rispetto al centro
            if cx < middle_x and cy < middle_y:
                return 'up-left'
            elif cx >= middle_x and cy < middle_y:
                return 'up-right'
            elif cx < middle_x and cy >= middle_y:
                return 'down-left'
            else:
                return 'down-right'
        
        for rect in sorted_rects:
            # Scegli la direzione preferita
            preferred_direction = choose_direction(rect)
            
            # Trova una posizione senza collisioni
            new_rect = LayoutManager.get_collision_free_position(
                rect, 
                positioned_rects, 
                fixed_zones=fixed_zones,
                preferred_direction=preferred_direction,
                padding=padding
            )
            
            # Mantieni l'elemento all'interno dello schermo
            margin = 5  # Margine dai bordi dello schermo
            if new_rect.right > screen_width - margin:
                new_rect.x = max(margin, screen_width - new_rect.width - margin)
            if new_rect.bottom > screen_height - margin:
                new_rect.y = max(margin, screen_height - new_rect.height - margin)
            if new_rect.x < margin:
                new_rect.x = margin
            if new_rect.y < margin:
                new_rect.y = margin
                
            # Aggiungi il nuovo rettangolo alla lista di quelli posizionati
            positioned_rects.append(new_rect)
            new_rects.append(new_rect)
        
        return new_rects