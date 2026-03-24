#!/usr/bin/env python3
"""Run the braiding annotation Flask app."""
from app import app

if __name__ == "__main__":
    with app.app_context():
        from models import db
        db.create_all()
        from models import User
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", is_admin=True)
            admin.set_password("admin")
            db.session.add(admin)
            db.session.commit()
            print("Created admin user: admin / admin")
    app.run(debug=True, host="0.0.0.0", port=5002, use_reloader=False)
