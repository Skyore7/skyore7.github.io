document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('csv-form');
    const responseDiv = document.getElementById('response-message');
    
    form.addEventListener('submit', function(event) {
        event.preventDefault();  // Prevent default form submission

        const fileInput = document.getElementById('csv-file');
        const file = fileInput.files[0];  // Get the selected file

        if (file) {
            // Create a FormData object to send the file
            const formData = new FormData();
            formData.append('csv-file', file);

            // Send the file via AJAX (using Fetch API)
            fetch('/process_csv', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())  // Expecting JSON response (x and y coordinates)
            .then(data => {
                // Extract x and y coordinates from the response
                const x = data.x;
                const y = data.y;

                // Display the coordinates for debugging (optional)
                responseDiv.innerHTML = `<p>Coordinates: x = ${x}, y = ${y}</p>`;

                // Get the player element and set its position
                const playerElement = document.getElementById('player0');
                if (playerElement) {
                    playerElement.style.position = 'absolute';  // Make sure it's positioned absolutely
                    playerElement.style.left = `${x}px`;       // Set the x coordinate
                    playerElement.style.top = `${y}px`;        // Set the y coordinate
                }
            })
            .catch(error => {
                responseDiv.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
            });
        }
    });
});
