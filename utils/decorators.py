from functools import wraps
from flask import session, redirect, url_for, flash, request
import time

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('⚠️ Please login to access this page', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('⚠️ Please login to access this page', 'warning')
            return redirect(url_for('auth.login'))
        
        # Check if user is admin (you can modify this logic)
        if not session.get('is_admin', False):
            flash('❌ Admin access required', 'error')
            return redirect(url_for('dashboard.dashboard_page'))
        
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(max_calls=10, time_window=60):
    """Simple rate limiting decorator"""
    calls = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = session.get('user_id', request.remote_addr)
            current_time = time.time()
            
            if user_id not in calls:
                calls[user_id] = []
            
            # Remove old calls outside time window
            calls[user_id] = [call_time for call_time in calls[user_id] 
                             if current_time - call_time < time_window]
            
            # Check if rate limit exceeded
            if len(calls[user_id]) >= max_calls:
                flash('⚠️ Rate limit exceeded. Please try again later.', 'warning')
                return redirect(url_for('dashboard.dashboard_page'))
            
            calls[user_id].append(current_time)
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator