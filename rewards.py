def compute_final_score_breakdown(game_state):
    """
    Come compute_final_score, ma restituisce un breakdown dettagliato 
    e un "total" per ciascuna squadra (0 e 1).
    """
    squads = game_state["captured_squads"]

    # Conteggio scope
    scope0 = 0
    scope1 = 0
    for move in game_state["history"]:
        if move["capture_type"] == "scopa":
            if move["player"] in [0,2]:
                scope0 += 1
            else:
                scope1 += 1

    # Carte totali
    c0 = len(squads[0])
    c1 = len(squads[1])
    pt_c0, pt_c1 = (1,0) if c0>c1 else (0,1) if c1>c0 else (0,0)

    # Denari
    den0 = sum(1 for c in squads[0] if c[1]=='denari')
    den1 = sum(1 for c in squads[1] if c[1]=='denari')
    pt_d0, pt_d1 = (1,0) if den0>den1 else (0,1) if den1>den0 else (0,0)

    # Settebello
    sb0 = 1 if (7,'denari') in squads[0] else 0
    sb1 = 1 if (7,'denari') in squads[1] else 0

    # Primiera
    val_map = {1:16,2:12,3:13,4:14,5:15,6:18,7:21,8:10,9:10,10:10}
    best0 = {"denari":0,"coppe":0,"spade":0,"bastoni":0}
    best1 = {"denari":0,"coppe":0,"spade":0,"bastoni":0}
    for (r,s) in squads[0]:
        v = val_map[r]
        if v>best0[s]:
            best0[s] = v
    for (r,s) in squads[1]:
        v = val_map[r]
        if v>best1[s]:
            best1[s] = v
    prim0 = sum(best0.values())
    prim1 = sum(best1.values())
    pt_p0, pt_p1 = (1,0) if prim0>prim1 else (0,1) if prim1>prim0 else (0,0)

    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1

    breakdown = {
      0: {
         "carte": pt_c0,
         "denari": pt_d0,
         "settebello": sb0,
         "primiera": pt_p0,
         "scope": scope0,
         "total": total0
      },
      1: {
         "carte": pt_c1,
         "denari": pt_d1,
         "settebello": sb1,
         "primiera": pt_p1,
         "scope": scope1,
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
