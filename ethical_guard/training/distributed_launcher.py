import os
import sys
import subprocess

class ClusterLauncher:
    def __init__(self, num_gpus: int, config_path: str):
        self.num_gpus = num_gpus
        self.config_path = config_path

    def launch_training_cluster(self, script_path: str, training_args: list):
        """
        Spawns distributed training tasks across available GPU architectures.
        """
        print(f"[INFO] Initializing multi-node topology via DeepSpeed. Workers: {self.num_gpus}")
        
        # Assemble standard PyTorch distributed launch components with DeepSpeed integration
        cmd = [
            "deepspeed",
            f"--num_gpus={self.num_gpus}",
            script_path,
            "--deepspeed", self.config_path
        ] + training_args

        # Execute environmental sub-process shell to monitor cluster TFLOPS output 
        try:
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"Distributed training crashed with exit code: {process.returncode}")
        except Exception as e:
            print(f"[CRITICAL] Operational cluster failure encountered: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    # Internal initialization test sequence
    launcher = ClusterLauncher(num_gpus=8, config_path="configs/ds_config_zero3.json")
    print("Distributed Infrastructure Cluster Launcher Core Operational.")