from flask import Flask, render_template, redirect, url_for
from config import Config
from database.db import db, init_db
from routes import auth, stock, prediction, dashboard, visuals, trending, chatbot

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Register blueprints
app.register_blueprint(auth.bp)
app.register_blueprint(stock.bp)
app.register_blueprint(prediction.bp)
app.register_blueprint(dashboard.bp)
app.register_blueprint(visuals.bp)
app.register_blueprint(trending.bp)
app.register_blueprint(chatbot.bp)

@app.route('/')
def index():
    """Home page - redirect to login"""
    return redirect(url_for('auth.login'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', error="404 - Page not found 😔"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('base.html', error="500 - Internal server error 🔥"), 500

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True, port=5000)