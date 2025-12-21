import subprocess
import sys

BIN = "./nested_rowminmax"


def run_case(args):
    proc = subprocess.run(
        [BIN] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def main() -> None:
    tests = [
        ["--nrows", "100", "--ncols", "100", "--mode", "outer", "--threads", "4"],
        ["--nrows", "200", "--ncols", "200", "--mode", "nested",
         "--outer_threads", "2", "--inner_threads", "2"],
    ]

    for idx, args in enumerate(tests, 1):
        code, out, err = run_case(args)
        print(f"[case {idx}] args:", " ".join(args))
        print("  exit code:", code)
        if out:
            print("  stdout:", out)
        if err:
            print("  stderr:", err)
        print()

    if any(run_case(a)[0] != 0 for a in tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
