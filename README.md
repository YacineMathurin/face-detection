Face Storage:

Reads faces directly from a directory structure
Organizes faces by badge ID
Supports multiple photos per person for better recognition


Requirements:

bashCopypip install opencv-python requests
To use this system:

Create a directory structure for known faces:

Copyfaces_directory/
    badge_id_1/
        photo1.jpg
        photo2.jpg
    badge_id_2/
        photo1.jpg
        ...

Configure the access system URL and authentication:

pythonCopyaccess_system_url = "https://your-access-system-url"
api_key = "your-api-key"  # if required
