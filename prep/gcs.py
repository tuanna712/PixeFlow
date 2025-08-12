import os
from google.cloud import storage
from google.api_core.exceptions import NotFound

class GCSManager:
    """
    A class to manage a Google Cloud Storage bucket with various file and folder operations.
    Authentication is handled via a service account key file.
    """
    def __init__(self, bucket_name, service_account_path):
        """
        Initializes the GCSManager.

        Args:
            bucket_name (str): The name of the Google Cloud Storage bucket.
            service_account_path (str): The path to the service account JSON key file.
        """
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                f"The service account key file was not found at: {service_account_path}"
            )
        
        try:
            self.storage_client = storage.Client.from_service_account_json(service_account_path)
            self.bucket = self.storage_client.bucket(bucket_name)
            print(f"Connected to bucket '{self.bucket.name}' successfully.")
        except Exception as e:
            print(f"Error connecting to Google Cloud Storage: {e}")
            self.storage_client = None
            self.bucket = None

    def get_all_folders(self):
        """
        Lists all folders in the bucket.

        Returns:
            list: A list of folder paths.
        """
        if not self.bucket:
            return []

        folders = set()
        try:
            blobs = self.bucket.list_blobs()
            for blob in blobs:
                folder = os.path.dirname(blob.name)
                if folder:
                    folders.add(folder)
            return list(folders)
        except Exception as e:
            print(f"Error getting all folders: {e}")
            return []

    def check_folder_exists(self, folder_path):
        """
        Checks if a folder path exists in a bucket by looking for objects
        that start with the given folder prefix.
        
        Args:
            folder_path (str): The path of the folder to check (e.g., 'images/').

        Returns:
            bool: True if the folder contains at least one object, False otherwise.
        """
        if not self.bucket:
            return False
            
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        try:
            blobs = self.bucket.list_blobs(prefix=folder_path, max_results=1)
            return any(blobs)
        except Exception as e:
            print(f"Error checking folder existence: {e}")
            return False

    def upload_file(self, source_file_path, destination_blob_name):
        """
        Uploads a file to the bucket.
        
        Args:
            source_file_path (str): The path to the local file.
            destination_blob_name (str): The desired name for the object in the bucket.
        
        Returns:
            str: The public URL of the uploaded file if successful, otherwise None.
        """
        if not self.bucket:
            return None
            
        if not os.path.exists(source_file_path):
            print(f"Error: Local file '{source_file_path}' not found.")
            return None
            
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)
            blob.make_public()
            public_url = blob.public_url
            print(f"File '{source_file_path}' uploaded to '{destination_blob_name}'.")
            return public_url
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def list_files(self, prefix=None):
        """
        Lists all files in the bucket or within a specified folder prefix.

        Args:
            prefix (str, optional): The folder prefix to list files from. Defaults to None.
        
        Returns:
            list: A list of blob names.
        """
        if not self.bucket:
            return []
            
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def download_file(self, source_blob_name, destination_file_path):
        """
        Downloads a file from the bucket.
        
        Args:
            source_blob_name (str): The name of the object in the bucket.
            destination_file_path (str): The local path to save the downloaded file.
        """
        if not self.bucket:
            return
            
        try:
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_path)
            print(f"File '{source_blob_name}' downloaded to '{destination_file_path}'.")
        except NotFound:
            print(f"Error: Blob '{source_blob_name}' not found.")
        except Exception as e:
            print(f"Error downloading file: {e}")

    def get_file_info(self, blob_name):
        """
        Fetches metadata for a single file without downloading it.

        Args:
            blob_name (str): The name of the object in the bucket.

        Returns:
            dict: A dictionary of file metadata if found, otherwise None.
        """
        if not self.bucket:
            return None
            
        try:
            blob = self.bucket.get_blob(blob_name)
            if blob:
                return {
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_type,
                    "media_link": blob.media_link,
                    "public_url": blob.public_url if blob.acl else "Not public"
                }
            else:
                print(f"Error: Blob '{blob_name}' not found.")
                return None
        except Exception as e:
            print(f"Error getting file info: {e}")
            return None

    def move_file(self, source_blob_name, destination_blob_name):
        """
        Moves a file by copying it to a new location and deleting the original.

        Args:
            source_blob_name (str): The name of the object to move.
            destination_blob_name (str): The new name for the object.
        """
        if not self.bucket:
            return
            
        try:
            source_blob = self.bucket.blob(source_blob_name)
            self.bucket.copy_blob(source_blob, self.bucket, destination_blob_name)
            source_blob.delete()
            print(f"File '{source_blob_name}' moved to '{destination_blob_name}'.")
        except NotFound:
            print(f"Error: Source blob '{source_blob_name}' not found.")
        except Exception as e:
            print(f"Error moving file: {e}")

    def copy_file(self, source_blob_name, destination_blob_name, destination_bucket_name=None):
        """
        Copies a file to a new location, optionally in a different bucket.

        Args:
            source_blob_name (str): The name of the object to copy.
            destination_blob_name (str): The new name for the object.
            destination_bucket_name (str, optional): The bucket to copy to. 
                                                     Defaults to the current bucket.
        """
        if not self.bucket:
            return
            
        try:
            source_blob = self.bucket.blob(source_blob_name)
            destination_bucket = self.bucket
            if destination_bucket_name:
                destination_bucket = self.storage_client.bucket(destination_bucket_name)
            
            self.bucket.copy_blob(source_blob, destination_bucket, destination_blob_name)
            print(f"File '{source_blob_name}' copied to '{destination_blob_name}'.")
        except NotFound:
            print(f"Error: Source blob '{source_blob_name}' not found.")
        except Exception as e:
            print(f"Error copying file: {e}")

    def delete_file(self, blob_name):
        """
        Deletes a single file from the bucket.

        Args:
            blob_name (str): The name of the object to delete.
        """
        if not self.bucket:
            return
            
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            print(f"File '{blob_name}' deleted.")
        except NotFound:
            print(f"Error: Blob '{blob_name}' not found.")
        except Exception as e:
            print(f"Error deleting file: {e}")

    def delete_folder(self, folder_path):
        """
        Deletes all files within a specified folder (prefix).

        Args:
            folder_path (str): The folder prefix to delete.
        """
        if not self.bucket:
            return
            
        try:
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            blobs_to_delete = list(self.bucket.list_blobs(prefix=folder_path))
            
            if not blobs_to_delete:
                print(f"Folder '{folder_path}' is empty or does not exist.")
                return

            self.bucket.delete_blobs(blobs_to_delete)
            print(f"Deleted all files in folder '{folder_path}'.")
        except Exception as e:
            print(f"Error deleting folder: {e}")

    def make_file_public(self, blob_name):
        """
        Makes a file publicly accessible.

        Args:
            blob_name (str): The name of the object to make public.
        """
        if not self.bucket:
            return
            
        try:
            blob = self.bucket.blob(blob_name)
            if blob:
                blob.make_public()
                print(f"File '{blob_name}' is now public.")
            else:
                print(f"Error: Blob '{blob_name}' not found.")
        except Exception as e:
            print(f"Error making file public: {e}")

    def make_file_private(self, blob_name):
        """
        Revokes public access to a file.

        Args:
            blob_name (str): The name of the object to make private.
        """
        if not self.bucket:
            return
            
        try:
            blob = self.bucket.blob(blob_name)
            if blob:
                blob.acl.all().grant_read() # This is one way to make it private, by granting specific permissions.
                blob.acl.save()
                print(f"File '{blob_name}' is now private.")
            else:
                print(f"Error: Blob '{blob_name}' not found.")
        except Exception as e:
            print(f"Error making file private: {e}")
            

"""
EXAMPLE USAGES
# ==== Upload file to blob ====
blob_name = os.path.normpath(frame_rel_path)
public_url = gcs_manager.upload_file(full_path, blob_name)
if public_url:
    print(f"Uploaded file is at: {public_url}")

# ==== Check if the folder exists ====
folder_check = gcs_manager.check_folder_exists("folder_name")
print(f"Does the folder exist? {folder_check}")

# ==== Get all folders ====
get_all_folders = gcs_manager.get_all_folders()
print(f"All folders in the bucket: {get_all_folders}")

# ==== Get file info, display and download url ====
file_info = gcs_manager.get_file_info(blob_name)
print("\nFile Info:")
if file_info:
    for key, value in file_info.items():
        print(f"  {key}: {value}")
# ==== Get all files =====
files_in_folder = gcs_manager.list_files()
print(len(files_in_folder))
for file in files_in_folder:
    print(f"  - {file}")

# ==== Delete files/ folder ======
gcs_manager.delete_file(blob_name)
gcs_manager.delete_folder("folder_name")
"""