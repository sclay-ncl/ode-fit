def ode_f(t, y0, params):
    
    a, b, p = y0

    dPdt = [-params["k"]*a*b, # A
            -params["k"]*a*b, # B
            params["k"]*a*b]  # P
    
    return dPdt