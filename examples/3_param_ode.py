def ode_f(t, y0, params):

    a, b, c, p = y0

    dPdt = [-params["k_plus"]*a*b + params["k_minus"]*c,                # A
            -params["k_plus"]*a*b + params["k_minus"]*c,                # B
            params["k_plus"]*a*b - (params["k_minus"] + params["k"])*c, # C
            params["k"]*c]                                              # P
    
    return dPdt
