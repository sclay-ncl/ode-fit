def ode_f(t, y0, params):
    
    invader1, invader2, gate, complex1, complex2, product = y0
    
    dPdt = [params["kh_minus"] * gate * complex1 - params["kh_plus"]* invader1 * gate, # invader1
            params["kh_minus"] * gate * complex2 - params["kh_plus"]* invader2 * gate, # invader2
            params["kh_minus"] * gate * (complex1 + complex2) - params["kh_plus"]*(invader1 + invader2) * gate, # gate
            params["kh_plus"] * invader1 * gate - params["kh_minus"] * gate * complex1 - params["kb1"] * complex1, # complex1
            params["kh_plus"] * invader2 * gate - params["kh_minus"] * gate * complex2 - params["kb2"] * complex2, #complex2
            params["kb1"] * complex1 + params["kb2"] * complex2 # product
            ]
    return dPdt