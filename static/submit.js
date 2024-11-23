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

                var player = 0;

                while (player < 6) {
                    // Get the player element and set its position
                    const playerElement = document.getElementById(`player${player}`);
                    if (playerElement) {
                        playerElement.style.position = 'absolute';  // Make sure it's positioned absolutely
                        playerElement.style.left = `${data[player][0]}px`;       // Set the x coordinate
                        playerElement.style.top = `${data[player][1]}px`;        // Set the y coordinate
                    }                   
                    
                    player += 1

                }
                const ballElement = document.getElementById(`ball`);
                ballElement.style.position = 'absolute';  // Make sure it's positioned absolutely
                ballElement.style.left = `${data[player][0]}px`;       // Set the x coordinate
                ballElement.style.top = `${data[player][1]}px`;        // Set the y coordinate



                
            })
            .catch(error => {
                responseDiv.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
            });
        }
    });
});
