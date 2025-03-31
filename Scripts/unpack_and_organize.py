import os
import shutil
import tarfile
import zipfile
import stat


class DataUnpacker:
    def __init__(self, packed_dir, unpacked_dir, temp_extract):
        self.packed_dir = packed_dir
        self.unpacked_dir = unpacked_dir
        self.temp_extract = temp_extract

        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.packed_dir, exist_ok=True)
        os.makedirs(self.unpacked_dir, exist_ok=True)
        os.makedirs(self.temp_extract, exist_ok=True)

    def extract_files(self, file_path, target_dir):
        if file_path.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
                print(f"✅ Extraído: {file_path} -> {target_dir}")
        elif file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(path=target_dir)
                print(f"✅ Extraído: {file_path} -> {target_dir}")

    def recursive_extract(self, folder):
        new_file = True
        while new_file:
            new_file = False
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith((".zip", ".tar.gz")):
                        new_file = True
                        file_path = os.path.join(root, file)
                        base_name = os.path.splitext(
                            os.path.splitext(file)[0]
                            )[0]
                        target_subdir = os.path.join(root, base_name)
                        os.makedirs(target_subdir, exist_ok=True)
                        self.extract_files(file_path, target_subdir)
                        os.remove(file_path)

    def handle_initial_files(self):
        initial_files = [
            f for f in os.listdir(self.packed_dir)
            if f.endswith((".zip", ".tar.gz"))
            ]
        for filename in initial_files:
            src_path = os.path.join(self.packed_dir, filename)
            self.extract_files(src_path, self.temp_extract)

    def move_content(self):
        for item in os.listdir(self.temp_extract):
            item_path = os.path.join(self.temp_extract, item)
            target_base = os.path.join(self.unpacked_dir, item)

            if os.path.isdir(item_path):
                if os.path.exists(target_base):
                    shutil.rmtree(target_base)
                shutil.move(item_path, target_base)
                print(f"📁 Pasta movida: {item_path} -> {target_base}")
            elif os.path.isfile(item_path):
                shutil.move(item_path, target_base)
                print(f"📄 Arquivo movido: {item_path} -> {target_base}")

    def _handle_remove_readonly(self, func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def clean_up(self):
        shutil.rmtree(self.temp_extract, onerror=self._handle_remove_readonly)
        print("🧹 Limpeza da pasta temporária concluída.")

    def execute(self):
        print("🚀 Iniciando organização dos dados...\n")
        self.handle_initial_files()
        self.recursive_extract(self.temp_extract)
        self.move_content()
        self.clean_up()
        print("\n✅ Processo finalizado com sucesso!")


# =======================
# Execução do processo
# =======================

if __name__ == "__main__":
    unpacker = DataUnpacker(
        packed_dir="Database/Arquive/",
        unpacked_dir="Database/",
        temp_extract="Database/temp_unpack/"
    )
    unpacker.execute()
