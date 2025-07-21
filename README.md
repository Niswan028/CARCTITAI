ere is the full sequence of steps to get your project working from start to finish.

Step 1: Get Your API Keys
You need two secret keys.

Groq API Key: Go to https://console.groq.com/keys, create a new key, and copy it.

Hugging Face Token: Go to https://huggingface.co/settings/tokens, create a new token, and make sure to give it write permissions. Copy this token.

Step 2: Set Up Your .env File
In your project folder, open the .env file and paste your Groq API Key into it.

GROQ_API_KEY="PASTE_YOUR_GROQ_API_KEY_HERE"

Step 3: Create and Use the login_helper.py File
This step will securely log you into Hugging Face.

In your main project folder (RAG_NCERT_BOT), create a new file named login_helper.py.

Copy and paste the code below into that file.

Replace PASTE_YOUR_NEW_HUGGING_FACE_WRITE_TOKEN_HERE with your Hugging Face Token.

from huggingface_hub import login

# Paste your Hugging Face token with WRITE permissions here
HF_TOKEN = "PASTE_YOUR_NEW_HUGGING_FACE_WRITE_TOKEN_HERE"

print("Attempting to log in to Hugging Face...")
try:
    login(token=HF_TOKEN, add_to_git_credential=True)
    print("\n✅ Login successful!")
except Exception as e:
    print(f"\n❌ Login failed. Error: {e}")

Now, run this script from your terminal:

python login_helper.py

Once you see "✅ Login successful!", you can delete the login_helper.py file.

Step 4: Install All Dependencies
Run this command in your terminal to install all the necessary Python packages.

pip install -r requirements.txt

Step 5: Build the Search Index
This command downloads the dataset from Hugging Face and builds the local search index.

python src/embedding_builder.py

Step 6: Run the Streamlit Application
This is the final step. It starts the web server and launches your chatbot.

streamlit run app.py

Your application should now be running correctly in your web browser.