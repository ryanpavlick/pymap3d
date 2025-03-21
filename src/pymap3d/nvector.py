import numpy as np
import pymap3d as pm

def geodetic2nvector(lat, lon, ell=None, deg=True):
    """
    Convert geodetic coordinates (latitude, longitude) to an n-vector.

    Parameters:
        lat : float or array-like
            Geodetic latitude(s).
        lon : float or array-like
            Geodetic longitude(s).
        ell : str or tuple, optional
            Reference ellipsoid (default is None, which uses WGS84).
        deg : bool, optional
            If True (default), inputs are in degrees. If False, use radians.

    Returns:
        n1, n2, n3 : ndarray
            Components of the n-vector in the Earth-Centered Earth-Fixed (ECEF) coordinate system.
    """
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)

    if deg:
        lat, lon = np.radians(lat), np.radians(lon)

    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    n1 = cos_lat * cos_lon
    n2 = cos_lat * sin_lon
    n3 = sin_lat

    return n1, n2, n3

def nvector2geodetic(n1, n2, n3, ell=None, deg=True):
    """
    Convert an n-vector back to geodetic coordinates (latitude, longitude).

    Parameters:
        n1, n2, n3 : float or array-like
            Components of the n-vector in the Earth-Centered Earth-Fixed (ECEF) coordinate system.
        ell : str or tuple, optional
            Reference ellipsoid (default is None, which uses WGS84).
        deg : bool, optional
            If True (default), returns latitude and longitude in degrees. If False, in radians.

    Returns:
        lat, lon : ndarray
            Geodetic latitude(s) and longitude(s).
    """
    n1, n2, n3 = np.atleast_1d(n1), np.atleast_1d(n2), np.atleast_1d(n3)

    # Compute latitude and longitude from n-vector
    lat = np.arcsin(n3)
    lon = np.arctan2(n2, n1)

    if deg:
        lat, lon = np.degrees(lat), np.degrees(lon)

    return lat, lon

def ecef2nvector(x, y, z, ell=None, deg=True):
    """
    Convert ECEF coordinates to an n-vector.

    Parameters:
        x, y, z : float or array-like
            ECEF coordinates in meters.
        ell : str or tuple, optional
            Reference ellipsoid (default is None, which uses WGS84).
        deg : bool, optional
            If True (default), geodetic2nvector() inputs are in degrees. If False, in radians.

    Returns:
        n1, n2, n3 : ndarray
            Components of the n-vector in the Earth-Centered Earth-Fixed (ECEF) coordinate system.
    """
    lat, lon, _ = pm.ecef2geodetic(x, y, z, ell=ell, deg=deg)
    return geodetic2nvector(lat, lon, ell=ell, deg=deg)

def nvector2ecef(n1, n2, n3, alt=0, ell=None, deg=True):
    """
    Convert an n-vector to ECEF coordinates.

    Parameters:
        n1, n2, n3 : float or array-like
            Components of the n-vector in the Earth-Centered Earth-Fixed (ECEF) coordinate system.
        alt : float or array-like, optional
            Altitude in meters (default is 0).
        ell : str or tuple, optional
            Reference ellipsoid (default is None, which uses WGS84).
        deg : bool, optional
            If True (default), nvector2geodetic() outputs are in degrees. If False, in radians.

    Returns:
        x, y, z : ndarray
            ECEF coordinates in meters.
    """
    lat, lon = nvector2geodetic(n1, n2, n3, ell=ell, deg=deg)
    return pm.geodetic2ecef(lat, lon, alt, ell=ell, deg=deg)
