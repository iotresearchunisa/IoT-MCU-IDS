# Accessing Google Drive via Python
This guide shows you how to generate the necessary tokens to access Google Drive via Python using Google Cloud APIs.

## Prerequisites
1. **Google Cloud Account**: make sure you have a Google Cloud account. If you don't have one, you can create it at [Google Cloud](https://cloud.google.com/).

2. **Google Cloud Project**: create a new project on [Google Cloud Console](https://console.cloud.google.com/).

3. **Python**: ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

## Steps to Configure Access to Google Drive
### 1. Create a CSV File on Google Drive
1. Open your Google Drive.
2. Click on **New** and select **Google Sheets**.
3. In the new Google Sheet, go to **File > Download > Comma-separated values (.csv, current sheet)**.
4. Save the file as `iot.csv` on your Google Drive.

### 2. Enable Google Drive API
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select your project.
3. In the navigation menu, go to **API & Services > Dashboard**.
4. Click on **Enable APIs and Services**.
5. Search for **Google Drive API** and enable it.

### 3. Create OAuth 2.0 Credentials
1. In the [Google Cloud Console](https://console.cloud.google.com/), go to **API & Services > Credentials**.
2. Click on **Create Credentials** and select **OAuth 2.0 Client ID**.
3. Configure the OAuth consent screen if required.
4. Choose **Application type** as **Desktop app** and give your credentials a name.
5. Click on **Create**.
6. Download the JSON file with the credentials and save it in a secure place.

### 4. Create a Service Account and Generate a Token
1. In the [Google Cloud Console](https://console.cloud.google.com/), go to **IAM & Admin > Service Accounts**.
2. Click on **Create Service Account**.
3. Provide a name and ID for the service account and click **Create**.
4. Assign the role **Editor** or **Owner** to the service account.
5. Click **Continue** and then **Done**.
6. In the Service Accounts page, click on the service account you created.
7. Go to the **Keys** tab and click **Add Key > Create New Key**.
8. Select **JSON** as the key type and click **Create**.
9. Download the JSON key file and save it in a secure place.

### 5. Share Google Drive Folder with Service Account
1. Open your Google Drive.
2. Right-click on the folder containing `iot.csv` and select **Share**.
3. In the **Share with people and groups** dialog, add the email address of your service account. The email will be in the form `your-service-account-name@your-project-id.iam.gserviceaccount.com`.
4. Set the permissions to **Editor**.
5. Click **Send**.

## Install Python Libraries
Make sure you have `pip` installed and use the following command to install the necessary libraries:

```sh
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## Placement of Credentials
The OAuth JSON file (credentials.json) should be placed in the `clent` folder. This is used by the client applications. 

The service account token (service-account-file.json) should be placed in the `server_raspi` folder. This is used by the Raspberry Pi. 
