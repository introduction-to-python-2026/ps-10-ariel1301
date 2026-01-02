import numpy as np
from PIL import Image
from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import disk

def main():
    # שם הקובץ בדיוק כפי שמופיע בתיקייה שלך
    path = 'original_image.png' 
    
    # טעינה
    image = load_image(path)
    
    # בדיקת תקינות - זה ימנע את השגיאה שקיבלת
    if image is None:
        print("Error: Could not find or load the image!")
        return

    # 1. המרה לגרייסקייל
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # 2. ניקוי רעשים (שימוש ב-disk עבור תמונה דו-מימדית)
    clean = median(gray.astype(np.uint8), disk(3))

    # 3. זיהוי קצוות
    edges = edge_detection(clean)

    # 4. בינאריזציה (Threshold) - בתמונה החדשה הרקע לבן, נסי 50
    edge_binary = edges > 50

    # 5. שמירה
    result = Image.fromarray((edge_binary * 255).astype(np.uint8))
    result.save('my_edges.png')
    print("Success! File 'my_edges.png' created.")

if __name__ == "__main__":
    main()
