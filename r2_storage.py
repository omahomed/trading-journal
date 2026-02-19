"""
Cloudflare R2 Storage Helper Functions
Handles image upload/download/delete for trade charts
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import streamlit as st
from datetime import datetime
import io
from typing import Optional, List, Dict


def get_r2_client():
    """
    Initialize and return R2 client using credentials from Streamlit secrets
    """
    try:
        r2_config = st.secrets.get("r2", {})

        if not r2_config:
            raise ValueError("R2 credentials not found in Streamlit secrets")

        endpoint_url = r2_config.get("endpoint_url")
        access_key_id = r2_config.get("access_key_id")
        secret_access_key = r2_config.get("secret_access_key")

        if not all([endpoint_url, access_key_id, secret_access_key]):
            raise ValueError("Missing R2 credentials: endpoint_url, access_key_id, or secret_access_key")

        client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='auto'  # Cloudflare R2 uses 'auto' for region
        )

        return client

    except Exception as e:
        st.error(f"Failed to initialize R2 client: {e}")
        return None


def upload_image(
    file_obj,
    portfolio_name: str,
    trade_id: str,
    ticker: str,
    image_type: str
) -> Optional[str]:
    """
    Upload an image to R2 storage

    Args:
        file_obj: Streamlit UploadedFile object or file-like object
        portfolio_name: Portfolio name (e.g., 'CanSlim')
        trade_id: Trade ID (e.g., 'NVDA-001')
        ticker: Stock ticker
        image_type: 'weekly', 'daily', or 'exit'

    Returns:
        R2 object key (path) if successful, None if failed
    """
    try:
        print(f"[R2] Starting upload: {image_type} for {trade_id}")

        client = get_r2_client()
        if not client:
            error_msg = "R2 client initialization failed - check credentials"
            print(f"[R2 ERROR] {error_msg}")
            st.error(error_msg)
            return None

        bucket_name = st.secrets.get("r2", {}).get("bucket_name")
        if not bucket_name:
            error_msg = "R2 bucket_name not found in secrets"
            print(f"[R2 ERROR] {error_msg}")
            st.error(error_msg)
            return None

        print(f"[R2] Using bucket: {bucket_name}")

        # Generate unique object key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = file_obj.name.split('.')[-1] if hasattr(file_obj, 'name') else 'png'

        # Format: portfolio/trade_id/image_type_timestamp.ext
        object_key = f"{portfolio_name}/{trade_id}/{image_type}_{timestamp}.{file_extension}"

        print(f"[R2] Object key: {object_key}")

        # Read file content
        file_obj.seek(0)  # Reset file pointer
        file_content = file_obj.read()

        print(f"[R2] File size: {len(file_content)} bytes")

        # Upload to R2
        print(f"[R2] Uploading to bucket...")
        client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=file_content,
            ContentType=f'image/{file_extension}',
            Metadata={
                'portfolio': portfolio_name,
                'trade_id': trade_id,
                'ticker': ticker,
                'image_type': image_type,
                'uploaded_at': timestamp
            }
        )

        print(f"[R2] Upload successful: {object_key}")
        return object_key

    except Exception as e:
        error_msg = f"Failed to upload image to R2: {type(e).__name__}: {str(e)}"
        print(f"[R2 ERROR] {error_msg}")
        st.error(error_msg)
        import traceback
        print(traceback.format_exc())
        return None


def download_image(object_key: str) -> Optional[bytes]:
    """
    Download an image from R2 storage

    Args:
        object_key: R2 object key (path)

    Returns:
        Image bytes if successful, None if failed
    """
    try:
        client = get_r2_client()
        if not client:
            return None

        bucket_name = st.secrets.get("r2", {}).get("bucket_name")
        if not bucket_name:
            return None

        response = client.get_object(Bucket=bucket_name, Key=object_key)
        return response['Body'].read()

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.warning(f"Image not found: {object_key}")
        else:
            st.error(f"Failed to download image: {e}")
        return None

    except Exception as e:
        st.error(f"Failed to download image: {e}")
        return None


def delete_image(object_key: str) -> bool:
    """
    Delete an image from R2 storage

    Args:
        object_key: R2 object key (path)

    Returns:
        True if successful, False if failed
    """
    try:
        client = get_r2_client()
        if not client:
            return False

        bucket_name = st.secrets.get("r2", {}).get("bucket_name")
        if not bucket_name:
            return False

        client.delete_object(Bucket=bucket_name, Key=object_key)
        return True

    except Exception as e:
        st.error(f"Failed to delete image from R2: {e}")
        return False


def list_images(portfolio_name: str, trade_id: str) -> List[Dict]:
    """
    List all images for a specific trade

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade ID

    Returns:
        List of image metadata dictionaries
    """
    try:
        client = get_r2_client()
        if not client:
            return []

        bucket_name = st.secrets.get("r2", {}).get("bucket_name")
        if not bucket_name:
            return []

        prefix = f"{portfolio_name}/{trade_id}/"

        response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        images = []
        if 'Contents' in response:
            for obj in response['Contents']:
                images.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })

        return images

    except Exception as e:
        st.error(f"Failed to list images from R2: {e}")
        return []


def get_image_url(object_key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for viewing an image

    Args:
        object_key: R2 object key (path)
        expiration: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL if successful, None if failed
    """
    try:
        client = get_r2_client()
        if not client:
            return None

        bucket_name = st.secrets.get("r2", {}).get("bucket_name")
        if not bucket_name:
            return None

        url = client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key
            },
            ExpiresIn=expiration
        )

        return url

    except Exception as e:
        st.error(f"Failed to generate presigned URL: {e}")
        return None


def delete_all_trade_images(portfolio_name: str, trade_id: str) -> bool:
    """
    Delete all images for a specific trade

    Args:
        portfolio_name: Portfolio name
        trade_id: Trade ID

    Returns:
        True if successful, False if failed
    """
    try:
        images = list_images(portfolio_name, trade_id)

        if not images:
            return True

        client = get_r2_client()
        if not client:
            return False

        bucket_name = st.secrets.get("r2", {}).get("bucket_name")
        if not bucket_name:
            return False

        # Delete each image
        for img in images:
            client.delete_object(Bucket=bucket_name, Key=img['key'])

        return True

    except Exception as e:
        st.error(f"Failed to delete trade images: {e}")
        return False
