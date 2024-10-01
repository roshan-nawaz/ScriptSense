# CSS for additional styling
CSS = """
<style>
body {
    background: white;
    color: black;
    font-family: 'Arial', Serif;
    text-align: justify;
}

h1, h2, h3, h4, h5, h6 {
    color: black;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    text-align: center;
}

.stButton>button {
    background-color: green;
    color: white;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    margin: 4px 2px;
    transition: 0.3s;
    cursor: pointer;
    border-radius: 4px;
}

.stButton>button:hover {
    color: green;
    background-color: white;
    border: 1px solid green;
}

.stFileUploader {
    padding: 10px 0;
}

.stTextInput>div>div>input {
    border: 1px solid white;
    border-radius: 4px;
    padding: 10px;
    text-align: left;
}

.stMarkdown h2 {
    border-bottom: 2px solid #ddd;
    padding-bottom: 10px;
}

.stMarkdown h2, .stMarkdown h3 {
    margin-top: 20px;
}

.stMarkdown div p {
    margin: 10px 0;
}

</style>
"""