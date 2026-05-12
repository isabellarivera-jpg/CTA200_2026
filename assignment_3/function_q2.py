def lorenz(t, W, p, r, b):
    """
    This function returns the time derivatives of the Lorenz system at a given state.

    Parameters
    ----------
    t : float
        Time (not used explicitly, required by solve_ivp).

    W : list
        State vector [x, y, z].

    p : float
        Prandtl number; ratio of kinematic viscosity to thermal diffusivity.

    r : float
        Rayleigh number; depends on the vertical temperature difference
        across the atmosphere.

    b : float
        Dimensionless length scale.

    Returns
    -------
    list
        Time derivatives [x_dot, y_dot, z_dot].
    """
    #unpack the state vector into individual variables
    x = W[0]
    y = W[1]
    z = W[2]

    #compute the three time derivatives from Lorenz's equations
    x_dot = -p * (x - y)
    y_dot = (r * x) - y - (x * z)
    z_dot = - (b * z) + (x * y)

    # return as a list so solve_ivp can read it as a state vector
    return [x_dot, y_dot, z_dot] 