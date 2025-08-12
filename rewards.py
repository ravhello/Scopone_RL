def compute_final_score_breakdown(game_state, rules=None):
    """
    Come compute_final_score, ma restituisce un breakdown dettagliato 
    e un "total" per ciascuna squadra (0 e 1).

    Parametri opzionali tramite "rules":
      - re_bello: bool
      - napola: bool
      - napola_scoring: "fixed3" | "length"
    """
    squads = game_state["captured_squads"]

    rules = rules or {}
    re_bello_enabled = bool(rules.get("re_bello", False))
    napola_enabled = bool(rules.get("napola", False))
    napola_scoring = rules.get("napola_scoring", "fixed3")

    # Conteggio scope
    scope0 = 0
    scope1 = 0
    for move in game_state["history"]:
        if move.get("capture_type") == "scopa":
            if move.get("player") in [0, 2]:
                scope0 += 1
            else:
                scope1 += 1

    # Carte totali
    c0 = len(squads[0])
    c1 = len(squads[1])
    pt_c0, pt_c1 = (1, 0) if c0 > c1 else (0, 1) if c1 > c0 else (0, 0)

    # Denari
    den0 = sum(1 for c in squads[0] if c[1] == 'denari')
    den1 = sum(1 for c in squads[1] if c[1] == 'denari')
    pt_d0, pt_d1 = (1, 0) if den0 > den1 else (0, 1) if den1 > den0 else (0, 0)

    # Settebello
    sb0 = 1 if (7, 'denari') in squads[0] else 0
    sb1 = 1 if (7, 'denari') in squads[1] else 0

    # Primiera
    val_map = {1: 16, 2: 12, 3: 13, 4: 14, 5: 15, 6: 18, 7: 21, 8: 10, 9: 10, 10: 10}
    best0 = {"denari": 0, "coppe": 0, "spade": 0, "bastoni": 0}
    best1 = {"denari": 0, "coppe": 0, "spade": 0, "bastoni": 0}
    for (r, s) in squads[0]:
        v = val_map[r]
        if v > best0[s]:
            best0[s] = v
    for (r, s) in squads[1]:
        v = val_map[r]
        if v > best1[s]:
            best1[s] = v
    prim0 = sum(best0.values())
    prim1 = sum(best1.values())
    pt_p0, pt_p1 = (1, 0) if prim0 > prim1 else (0, 1) if prim1 > prim0 else (0, 0)

    # Varianti opzionali
    rb0 = 1 if re_bello_enabled and (10, 'denari') in squads[0] else 0
    rb1 = 1 if re_bello_enabled and (10, 'denari') in squads[1] else 0

    def napola_points(cards):
        if not napola_enabled:
            return 0
        denari_ranks = {r for (r, s) in cards if s == 'denari'}
        # Richiede almeno A-2-3
        if not {1, 2, 3}.issubset(denari_ranks):
            return 0
        if napola_scoring == "length":
            length = 0
            r = 1
            while r in denari_ranks:
                length += 1
                r += 1
            return length
        # default: fixed3
        return 3

    np0 = napola_points(squads[0])
    np1 = napola_points(squads[1])

    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0 + rb0 + np0
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1 + rb1 + np1

    breakdown = {
      0: {
         "carte": pt_c0,
         "denari": pt_d0,
         "settebello": sb0,
         "primiera": pt_p0,
         "scope": scope0,
         "re_bello": rb0 if re_bello_enabled else 0,
         "napola": np0 if napola_enabled else 0,
         "total": total0
      },
      1: {
         "carte": pt_c1,
         "denari": pt_d1,
         "settebello": sb1,
         "primiera": pt_p1,
         "scope": scope1,
         "re_bello": rb1 if re_bello_enabled else 0,
         "napola": np1 if napola_enabled else 0,
         "total": total1
      }
    }
    return breakdown

def compute_final_reward_from_breakdown(breakdown):
    """
    Calcola la differenza tra breakdown[0]["total"] e breakdown[1]["total"] * 10
    """
    diff = breakdown[0]["total"] - breakdown[1]["total"]
    return {
        0: diff*10,
        1: -diff*10
    }
