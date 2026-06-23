from abc import ABC, abstractmethod

class BaseSegmenter(ABC):
    """
    The absolute contract for all segmentation models in AVSplat-Sim.
    Any new segmenter added to the project MUST inherit from this class
    and implement the extract_masks method.
    """
    
    @abstractmethod
    def extract_masks(self, input_dir: str, output_dir: str, prompt: str):
        """
        Extracts segmentation masks from a directory of images.
        
        Args:
            input_dir (str): Absolute path to the folder containing raw images.
            output_dir (str): Absolute path to save the generated masks.
            prompt (str): The text or conditional prompt to guide the segmentation.
            
        Raises:
            NotImplementedError: If the child class fails to define this method.
        """
        pass
    
    
class BaseDiffusionModel(ABC):
    """
    The absolute contract for all diffusion models in AVSplat-Sim.
    Any new diffusion model added to the project MUST inherit from this class
    and implement the process_frames method.
    """
    
    @abstractmethod
    def process_frames(self, input_dir: str, output_dir: str, prompt: str):
        """
        Processes a directory of images using a diffusion model.
        
        Args:
            input_dir (str): Absolute path to the folder containing raw images.
            output_dir (str): Absolute path to save the processed images.
            prompt (str): The text or conditional prompt to guide the diffusion process.
            
        Raises:
            NotImplementedError: If the child class fails to define this method.
        """
        pass


class BaseTracker(ABC):
    """
    The absolute contract for all tracking models in AVSplat-Sim.
    Any new tracker added to the project MUST inherit from this class
    and implement the run_tracking method.
    """

    @abstractmethod
    def run_tracking(
        self,
        data_root: str,
        output_dir: str,
        cameras: list[str],
        task_cfg: dict,
    ) -> None:
        """
        Runs multi-camera tracking for a dataset scene.

        Args:
            data_root (str): Absolute path to the scene root directory.
            output_dir (str): Absolute path where tracking outputs are written.
            cameras (list[str]): Camera names to include in tracking.
            task_cfg (dict): Task-level runtime settings (mode, image_size, vis, etc).

        Raises:
            NotImplementedError: If the child class fails to define this method.
        """
        pass
    