const form = document.querySelector('form');

form.addEventListener('submit', async (event) => {
  event.preventDefault(); // Prevent page refresh

  const fileInput = document.getElementById('file');
  const file = fileInput.files[0]; // Get the selected file

  if (file) {
    try {
      const formData = new FormData();
      formData.append('file', file); // Add the file to the FormData

      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('File uploaded successfully!');
      } else {
        console.error('Error uploading file.');
      }
    } catch (error) {
      console.error('An error occurred:', error);
    }
  }
});


// Set up storage for uploaded files
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/'); // Specify the directory where files will be saved
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname); // Generate a unique filename
    },
});