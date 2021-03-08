import flask
from wild_sight import db

auth = flask.Blueprint("auth", __name__)


@auth.route("/login")
def login():
    return flask.render_template("profile.html")


@auth.route("/signup")
def signup():
    return flask.render_template("signup.html")


@auth.route("/logout")
def logout():
    return "Logout"
