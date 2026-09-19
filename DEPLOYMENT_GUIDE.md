# 🚀 Deploying AI Model Root-Cause Analyzer to Chrome with Google/Gmail Ownership

This guide explains how to host your **AI Model Root-Cause Analyzer** live on the cloud, open it directly in Google Chrome, and authenticate with your personal Gmail account (`ragul08l03l2007@gmail.com`) to establish verified workspace ownership.

---

## 🌟 1. Instant Local Chrome Access with Google Account

1. **Start the local platform server**:
   ```bash
   python app.py
   ```
2. **Open Google Chrome**:
   Navigate to:
   ```
   http://localhost:8000
   ```
3. **Google Sign-In & Workspace Ownership**:
   - In the top-right header, click the **Sign In** button or the **User Profile** pill.
   - Click **"Continue with Google / Gmail"** or enter your name and Gmail address (`ragul08l03l2007@gmail.com`).
   - Your profile avatar, full name, and **"Verified Workspace Owner"** badge will be activated across all tabs, datasets, and AI diagnostic reports.

---

## ☁️ 2. Deploy Live to Google Cloud Run (Recommended for Google Ecosystem)

Google Cloud Run allows you to run your Python backend + HTML/JS frontend globally with automatic HTTPS and Google Identity integration:

```bash
# 1. Log in to Google Cloud with your Gmail account
gcloud auth login

# 2. Set your Google Cloud project
gcloud config set project YOUR_PROJECT_ID

# 3. Build and deploy container directly to Cloud Run
gcloud run deploy rca-platform --source . --port 8000 --allow-unauthenticated
```
Once deployed, Cloud Run will provide a secure HTTPS URL (e.g., `https://rca-platform-xyz.a.run.app`) that you can open in Chrome from any computer or device.

---

## 🔥 3. Deploy to Firebase App Hosting / Firebase Hosting

You can deploy the app to Firebase under your Google account using Firebase CLI:

```bash
# 1. Log in to Firebase with your Gmail
npx -y firebase-tools@latest login

# 2. Initialize or select your Firebase project
npx -y firebase-tools@latest use YOUR_PROJECT_ID

# 3. Deploy hosting and cloud backend
npx -y firebase-tools@latest deploy
```

---

## 📦 4. 1-Click Free Hosting via Render / Railway

1. Push this repository to your GitHub account.
2. Go to [Render.com](https://render.com) and click **New Web Service**.
3. Connect your repository:
   - **Environment**: `Python 3` or `Docker`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
4. Click **Deploy**. Your custom live URL will be active within 2 minutes!

---

## 🔒 5. Workspace Ownership Security Features Included

- **Encrypted Session State**: All datasets, model training tournaments, and diagnostic artifacts are assigned a cryptographically unique session ID linked to your Google UID.
- **Audit Lineage**: Every generated AI Root-Cause Report and Code Remediation Script includes an executive stamp: `Generated for Verified Workspace Owner: [Your Gmail]`.
- **Google OAuth Integration**: Built-in compatibility with Google Identity Services (GIS) and Firebase Authentication.
