import numpy as np

def iterate(x, y, max_iter=100, threshold=2):
    """
    Iterates z_{i+1} = z_i^2 + c for each point c = x + iy in the complex plane
    and determines whether each point diverges or remains bounded.

    Parameters
    ----------
    x : np.ndarray
        2D array of real parts of c; spanning -2 to 2.
        
    y : np.ndarray
        2D array of imaginary parts of c; spanning -2 to 2.
        
    max_iter : int, optional
        Maximum number of iterations before a point is considered bounded.
        Default is 100.
        
    threshold : float, optional
        Value of |z| beyond which a point is considered diverged.
        Default is 2.

    Returns
    -------
    diverged : np.ndarray of bool
        2D boolean array, True where the point diverged, False where bounded.
        
    iter_count : np.ndarray of int
        2D array recording the iteration number at which each point diverged.
        Points that never diverge are assigned the value max_iter.

    """
    
    #defining complex plane equation
    c = x + (1j * y)
    
    #initializing
    z = np.zeros_like(c)
    
    #assume all points bounded (max_iter); update only if escapes
    iter_count = np.full(c.shape, max_iter, dtype=int)
    
    #tracking whether each point has escaped; all start as False
    diverged = np.zeros(c.shape, dtype=bool)
    
    
    #iterating through
    for i in range(max_iter):
        
        #only looking at points that have NOT escaped yet
        mask = ~diverged
        
        #applying the iteration
        z[mask] = z[mask]**2 + c[mask]
        
        #finding which ones escaped for this iteration
        newly_diverged = mask & (np.abs(z) > threshold)
        
        #noting down the iteration in which it diverged
        iter_count[newly_diverged] = i
        
        #merging the newly escaped ones into the masterlist of all that diverged
        diverged |= newly_diverged

    return diverged, iter_count