Execute the following steps in order (where `Exp_Conf.py` serves as the experimental configuration management file):

1. **Multi-channel Feature Extraction** (MF folder)  
   `pro_ohtertoken.py` → `pro_tldfea.py` → `pro_wordtoken.py`  

2. **PMAE Model** (PMAE folder)  
   `pretrained_MAE.py` → `test_bash` (execute the script to obtain optimal weights) → `pmae.py`  

---

### **Key Terminology**  
- **MF(Multi-channel Feature Extraction)**: Refers to the process of extracting heterogeneous features (e.g., TLD preferences, lexical patterns).  
- **PMAE (Pre-trained Multi-channel Autoencoder)**: The proposed model architecture.  
- **Optimal Weights**: The best-performing model parameters obtained through validation.  

### **Notes**  
- Ensure all dependencies are installed before execution.  
- Final evaluation is performed in `pmae.py` using the optimized weights.  

