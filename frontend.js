function search() {
    const query = document.getElementById('searchInput').value;
    const num = 10; // Number of recommendations, you can adjust as needed
    // Check if query is not empty
    if (!query.trim()) {
        alert("Please enter a search query");
        return;
    }
    document.getElementById('loading').style.display = 'block';

    // Show buffer message
    buffer.style.display = 'block';
    // Make a request to the backend with the query parameter and number of recommendations
    fetch(`http://127.0.0.1:5000/search?query=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            // Hide buffer message
            buffer.style.display = 'none';
            document.getElementById('loading').style.display = 'none';
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            // Hide buffer message
            document.getElementById('loading').style.display = 'none';
            buffer.style.display = 'none';
        });
}

function displayResults(results) {
    const searchResultsDiv = document.getElementById('searchResults');

    // Clear previous search results
    searchResultsDiv.innerHTML = '';

    // Display the name and link of the searched song
    const searchedSongDiv = document.createElement('div');
    searchedSongDiv.classList.add('searched-song');

    const searchedSongName = document.createElement('p');
    searchedSongName.textContent = `Searched Song: ${results.track_name}`;
    searchedSongDiv.appendChild(searchedSongName);

    const searchedSongLink = document.createElement('a');
    searchedSongLink.textContent = 'Listen on Spotify';
    searchedSongLink.href = results.link;
    searchedSongLink.target = '_blank'; // Open link in a new tab
    searchedSongDiv.appendChild(searchedSongLink);

    // Append the searched song div to the search results div
    searchResultsDiv.appendChild(searchedSongDiv);

    // Check if recommendations are available
    if (results && results.recommendations) {
        // Loop through each recommendation and create HTML elements to display them
        results.recommendations.forEach(recommendation => {
            const recommendationDiv = document.createElement('div');
            recommendationDiv.classList.add('recommendation');

            const trackName = document.createElement('p');
            trackName.textContent = recommendation.track_name;
            recommendationDiv.appendChild(trackName);

            const trackLink = document.createElement('a');
            trackLink.textContent = 'Listen on Spotify';
            trackLink.href = recommendation.link;
            trackLink.target = '_blank'; // Open link in a new tab
            recommendationDiv.appendChild(trackLink);

            // Append the recommendation div to the search results div
            searchResultsDiv.appendChild(recommendationDiv);
        });
    } else {
        // If no recommendations are available, display a message
        const noResultsMessage = document.createElement('p');
        noResultsMessage.textContent = 'No recommendations found.';
        searchResultsDiv.appendChild(noResultsMessage);
    }
}


