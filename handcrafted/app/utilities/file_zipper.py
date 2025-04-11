"""Module for zipping and unzipping files."""

import os
import zipfile


class FileZipper:
    """Class for zipping and unzipping files."""

    @staticmethod
    def zip_file(input_filename, output_filename=None):
        """Zip a file.

        Parameters
        ----------
        input_filename : str
            The name of the file to zip.
        output_filename : str, optional
            The name of the output zip file. If None, the input filename with .zip extension will be used.

        """
        if not output_filename:
            output_filename = input_filename + ".zip"

        with zipfile.ZipFile(
            output_filename, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            zip_file.write(
                input_filename, arcname=os.path.basename(input_filename)
            )

    @staticmethod
    def unzip_file(zip_filename, output_directory="."):
        """Unzip a file.

        Parameters
        ----------
        zip_filename : str
            The name of the zip file to unzip.
        output_directory : str, optional
            The directory where the files will be extracted. Defaults to the current directory.

        """
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(output_directory)
