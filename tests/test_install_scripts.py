import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class InstallScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="venice install ")
        self.addCleanup(self.temp_dir.cleanup)
        self.temp = Path(self.temp_dir.name)
        self.home = self.temp / "home with spaces"
        self.data_home = self.temp / "data with spaces"
        self.home.mkdir()
        self.data_home.mkdir()

        system_readlink = shutil.which("readlink")
        if system_readlink is None:
            self.skipTest("readlink is unavailable")
        shim_dir = self.temp / "bsd bin"
        shim_dir.mkdir()
        readlink_shim = shim_dir / "readlink"
        readlink_shim.write_text(
            "#!/bin/sh\n"
            "if [ \"${1-}\" = -f ]; then\n"
            "    echo 'readlink: illegal option -- f' >&2\n"
            "    exit 1\n"
            "fi\n"
            f"exec {self._shell_quote(system_readlink)} \"$@\"\n",
            encoding="utf-8",
        )
        readlink_shim.chmod(0o755)

        self.env = os.environ.copy()
        self.env.update(
            {
                "HOME": str(self.home),
                "XDG_DATA_HOME": str(self.data_home),
                "PATH": f"{shim_dir}{os.pathsep}{self.env.get('PATH', '')}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "VENICE_API_KEY": "test-fake-key",
            }
        )

        self.completion = (
            self.data_home / "bash-completion" / "completions" / "venice"
        )

    @staticmethod
    def _shell_quote(value):
        return "'" + value.replace("'", "'\\''") + "'"

    def run_script(self, name, *, through_symlink=False):
        script = ROOT / name
        if through_symlink:
            launcher_dir = self.temp / "launchers with spaces"
            launcher_dir.mkdir(exist_ok=True)
            absolute_link = launcher_dir / f"absolute-{name}"
            absolute_link.symlink_to(script)
            launcher = launcher_dir / name
            launcher.symlink_to(os.path.relpath(absolute_link, launcher_dir))
            script = launcher
        return subprocess.run(
            [str(script)],
            cwd=self.temp,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def test_bsd_readlink_supports_symlinked_install_and_uninstall(self):
        installed = self.run_script("install.sh", through_symlink=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)

        bin_link = self.home / ".local/bin/venice"
        lib_link = self.home / ".local/lib/venice"
        self.assertTrue(bin_link.is_symlink())
        self.assertTrue(lib_link.is_symlink())
        self.assertEqual(os.readlink(bin_link), str(ROOT / "bin/venice"))
        self.assertEqual(os.readlink(lib_link), str(ROOT / "src/venice"))
        config_dir = self.home / ".config/venice"
        self.assertEqual(stat.S_IMODE(config_dir.stat().st_mode), 0o700)
        self.assertEqual(
            self.completion.read_text(encoding="utf-8").splitlines()[0],
            f"# venice source completion owner: {ROOT}",
        )

        credentials = config_dir / "credentials"
        credentials.write_text("test-fake-key\n", encoding="utf-8")
        credentials.chmod(0o600)

        uninstalled = self.run_script("uninstall.sh", through_symlink=True)
        self.assertEqual(uninstalled.returncode, 0, uninstalled.stderr)
        self.assertFalse(bin_link.exists())
        self.assertFalse(lib_link.exists())
        self.assertFalse(self.completion.exists())
        self.assertTrue(credentials.exists())
        self.assertNotIn("shred", uninstalled.stdout)
        self.assertIn("rm -f ~/.config/venice/credentials", uninstalled.stdout)
        self.assertIn("rmdir ~/.config/venice", uninstalled.stdout)

    def test_uninstall_preserves_foreign_completion_with_registration(self):
        self.completion.parent.mkdir(parents=True)
        content = (
            "# user-managed completion\n"
            "_venice() { :; }\n"
            "complete -F _venice venice\n"
        )
        self.completion.write_text(content, encoding="utf-8")

        uninstalled = self.run_script("uninstall.sh")
        self.assertEqual(uninstalled.returncode, 0, uninstalled.stderr)
        self.assertEqual(self.completion.read_text(encoding="utf-8"), content)
        self.assertIn(f"skip     {self.completion} (not ours)", uninstalled.stdout)

    def test_uninstall_preserves_completion_owned_by_another_checkout(self):
        self.completion.parent.mkdir(parents=True)
        content = (
            f"# venice source completion owner: {self.temp / 'other checkout'}\n"
            "complete -F _venice venice\n"
        )
        self.completion.write_text(content, encoding="utf-8")

        uninstalled = self.run_script("uninstall.sh")
        self.assertEqual(uninstalled.returncode, 0, uninstalled.stderr)
        self.assertEqual(self.completion.read_text(encoding="utf-8"), content)

    def test_uninstall_preserves_completion_symlink(self):
        self.completion.parent.mkdir(parents=True)
        target = self.temp / "foreign completion"
        content = (
            f"# venice source completion owner: {ROOT}\n"
            "complete -F _venice venice\n"
        )
        target.write_text(content, encoding="utf-8")
        self.completion.symlink_to(target)

        uninstalled = self.run_script("uninstall.sh")
        self.assertEqual(uninstalled.returncode, 0, uninstalled.stderr)
        self.assertTrue(self.completion.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), content)

    def test_install_refuses_to_replace_a_regular_file(self):
        bin_dst = self.home / ".local/bin/venice"
        bin_dst.parent.mkdir(parents=True)
        bin_dst.write_text("user managed\n", encoding="utf-8")

        installed = self.run_script("install.sh")
        self.assertNotEqual(installed.returncode, 0)
        self.assertEqual(bin_dst.read_text(encoding="utf-8"), "user managed\n")
        self.assertIn("exists and is not a symlink", installed.stderr)

    def test_uninstall_preserves_links_that_point_elsewhere(self):
        bin_dst = self.home / ".local/bin/venice"
        lib_dst = self.home / ".local/lib/venice"
        bin_dst.parent.mkdir(parents=True)
        lib_dst.parent.mkdir(parents=True)
        bin_dst.symlink_to(self.temp / "foreign bin")
        lib_dst.symlink_to(self.temp / "foreign lib")

        uninstalled = self.run_script("uninstall.sh")
        self.assertEqual(uninstalled.returncode, 0, uninstalled.stderr)
        self.assertTrue(bin_dst.is_symlink())
        self.assertTrue(lib_dst.is_symlink())
        self.assertEqual(uninstalled.stdout.count("points elsewhere"), 2)


if __name__ == "__main__":
    unittest.main()
