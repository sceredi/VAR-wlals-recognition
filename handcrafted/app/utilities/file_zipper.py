import zipfile
import os


class FileZipper:
    @staticmethod
    def zip_file(input_filename, output_filename=None):
        if not output_filename:
            output_filename = input_filename + ".zip"

        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(input_filename, arcname=os.path.basename(input_filename))

    @staticmethod
    def unzip_file(zip_filename, output_directory="."):
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
