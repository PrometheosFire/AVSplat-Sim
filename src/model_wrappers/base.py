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
    