from selfplay.league import League


def test_league_register_and_sample():
    league = League(base_dir='checkpoints/league')
    # register fake paths (sampling doesn't check existence for Elo sampling)
    a = 'checkpoints/a.pth'
    b = 'checkpoints/b.pth'
    # mock: simulate file presence
    import os
    os.makedirs('checkpoints', exist_ok=True)
    open(a, 'a').close()
    open(b, 'a').close()
    league.register(a)
    league.register(b)
    p, o = league.sample_pair()
    assert p is not None and o is not None

