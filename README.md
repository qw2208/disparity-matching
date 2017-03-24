# disparity-matching
  

**In this project, we calculate the disparity map of two rectified images which imitate two eyes. In this project, SAD and DP algorithms (simplified) are implemented. PyCUDA codes are typically included.**     
1. "cudadpAdvanced.py" is the pycuda code for Dynamic Programming.  
2. "sad.py" is the pycuda code for local algorithms (SAD&SSD).  
3. "stereo-matching.py" is local algorithms which could be implemented on CPU.  
4. "dpmatching.py" is dp algorithm which could be implemented on CPU.  
5. "imL.png" and "imR.png" are input left images and right images.    

Execution of dp algorithm and sad are around 0.2-0.4s on GPU. We also enclosed disparity maps we derived.
      
`This is a code by Qingwei Wu and Reixuan Zhang` 
> Please feel free to contact me: qw2208@columbia.edu
