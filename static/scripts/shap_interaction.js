document.addEventListener('DOMContentLoaded', function () {
    var fileInput = document.getElementById('data_file');
    var uploadButton = document.getElementById('upload-file-button');
    var analysisButton = document.getElementById('shap-analysis-button');

    // Function to handle file upload asynchronously and display file data
    function handleFileUpload(event) {
        event.preventDefault(); // Prevent traditional form submission

        const file = fileInput.files[0];
        if (!file) {
            event.preventDefault();  // Stop form submission
            alert('Please select a file before uploading.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload_file', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Upload successful', data);
            displayFileData(file); // Call to display the file data
        })
        .catch(error => {
            console.error('Upload failed', error);
            alert('Failed to upload file. Check console for errors.');
        });
    }

    // Function to display file data after upload
function displayFileData(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        const allLines = content.split(/\r\n|\n/);
        const table = document.createElement('table');
        table.className = 'data';

        // Adding header row
        const headers = allLines[0].split(',');
        const headerRow = table.insertRow(-1);
        headers.forEach(header => {
            const headerCell = document.createElement("th");
            headerCell.innerText = header;
            headerRow.appendChild(headerCell);
        });

        // Adding data rows
        allLines.slice(1).forEach(line => {
            const row = table.insertRow(-1);
            line.split(',').forEach(cellText => {
                const cell = row.insertCell(-1);
                cell.innerText = cellText;
            });
        });

        const outputDiv = document.getElementById('file-content');
         if (outputDiv) {
            outputDiv.innerHTML = ''; // Clear previous content
            outputDiv.appendChild(table); // Append the new table
        } else {
            console.error('Failed to find the container to display the uploaded data.');
        }
    };
    reader.readAsText(file);
}

    // Function to handle SHAP Analysis submission
function submitSHAPForm() {
   console.log("Submitting SHAP form");
   const customerIdInput = document.getElementById('customer_id');
    if (!customerIdInput.value) {
        alert('Please enter a customerId for analysis.');
        return;
    }

    const data = { customer_id: customerIdInput.value };

    fetch('/perform_shap_analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Network response was not ok:${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
    const shapContainer = document.getElementById('shap-result-container');
    if (data.shapImageUrl && shapContainer) {
        const newPlotDiv = document.createElement('div');
        newPlotDiv.innerHTML = `<h3>Analysis for Customer ID: ${customerIdInput.value}</h3><iframe src="${data.shapImageUrl}" style="width:100%; height:600px; border:none;"></iframe>`;
        shapContainer.appendChild(newPlotDiv);  // Append the new plot to the container
    } else {
        console.error("shap-result-container not found or no image URL.");
        alert('Failed to perform SHAP analysis. Check console for errors.');
    }
  })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to perform SHAP analysis. Check console for errors.');
    });
}

    // Bind the upload and analysis functions to buttons
    if (uploadButton) {
        uploadButton.addEventListener('click', handleFileUpload);
    }

    if (analysisButton) {
        analysisButton.addEventListener('click', submitSHAPForm);
    }
});
