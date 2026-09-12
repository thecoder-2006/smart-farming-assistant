from app import app, load_model

load_model()

if __name__ == "__main__":
    app.run(debug=False)
