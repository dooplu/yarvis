

def lerp(a, b, t):
    '''
        linear interpolation function for smoothing
        
            Parameters:
                    a : start
                    b : end
                    t (float) : a value between 1 and 0

            Returns:
                    c (float) : a value t percent between a and b
    '''
    if t > 1: t = 1
    elif t < 0: t = 0
    c = (b-a)*t + a
    return c

def middle_point(a, b):
    x = (a[0] + b[0])/2.0
    y = (a[1] + b[1])/2.0
    return (x, y)
