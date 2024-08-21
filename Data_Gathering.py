import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

drug_data = pd.read_csv("/Users/kangaroo/Documents/Sydney_Uni_Study/2024/Honours/Semester_2/Data/Missed_DILIst_Drugs.csv")
drug_names = drug_data['Missed_Drugs'].tolist()

# Use a subset of 5 drug names for testing
drug_names_subset = drug_names[:5]

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# Function to get SMILES structure from PubChem
def get_smiles(drug_name):
    url = "https://pubchem.ncbi.nlm.nih.gov/"
    driver.get(url)
    
    # Locate the search bar using the updated ID
    search_bar = driver.find_element(By., "search_1722398967844")
    search_bar.clear()
    search_bar.send_keys(drug_name)
    search_bar.send_keys(Keys.RETURN)
    
    # Wait for search results to load
    time.sleep(3)
    
    try:
        # Click on the first result link
        first_result = driver.find_element(By.CSS_SELECTOR, ".search-result-title a")
        first_result.click()
        
        # Wait for the compound page to load
        time.sleep(3)
        
        # Get the SMILES structure
        smiles = driver.find_element(By.XPATH, "//div[text()='SMILES']/following-sibling::div").text
    except Exception as e:
        smiles = None
        print(f"Error retrieving SMILES for {drug_name}: {e}")
    
    return smiles

# Dictionary to store the results
results = {"Drug Name": [], "SMILES": []}

# Iterate through the list of drug names and get SMILES structures
for drug in drug_names:
    print(f"Fetching SMILES for {drug}...")
    smiles = get_smiles(drug)
    results["Drug Name"].append(drug)
    results["SMILES"].append(smiles)

# Convert results to a DataFrame and save as CSV
df = pd.DataFrame(results)
df.to_csv("drug_smiles.csv", index=False)

# Close the WebDriver
driver.quit()



