
import numpy.polynomial.polynomial as poly

def baseline_corrector(x,y,start_points=5,end_points=5,ax=None):
    """
    Transform data so distance between top and bottom baselines is 1.0
    across whole experiment. Fits one line to the start baseline, a 
    second line to the end baseline, and then transforms data to keep 
    the distance between those baselines 1.0. 
    
    y_out = (y - end_line)/(start_line - end_line)

    Parameters
    ----------
    x : numpy.ndarray
        float array of x values
    y : numpy.ndarray
        float array of y values
    start_points : int, default=5
        take first start_points of the experiment for the start baseline fit
    end_points : int, default=5
        take the last end_points of the experiment for the end baseline fit
    ax : maplotlib.Axes, optional
        if specified, plot the raw data and baselines on this axis
    """
    
    # Fit line to first start_points points on curve
    start_coefs = poly.polyfit(x[:start_points], y[:start_points], 2)
    start_b = start_coefs[0]
    start_m = start_coefs[1]
    start_line = start_m*x + start_b
    
    # Fit line to last end_points points on curve
    end_coefs = poly.polyfit(x[-end_points:], y[-end_points:], 2)
    end_b = end_coefs[0]
    end_m = end_coefs[1]
    end_line = end_m*x + end_b

    # Plot if an ax object is sent in
    if ax is not None:
        ax.scatter(x,y,s=50,facecolor='none',edgecolor='black')
        ax.plot(x,poly.polyval(x, start_coefs),'-',color="blue")
        ax.plot(x,poly.polyval(x, end_coefs),'-',color="red")

    return (y - end_line)/(start_line - end_line)
