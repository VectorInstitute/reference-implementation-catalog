document.addEventListener('DOMContentLoaded', function() {
  // Find all bibtex buttons
  const bibtexButtons = document.querySelectorAll('.bibtex-button');

  // Create modal container if it doesn't exist
  let modal = document.getElementById('bibtex-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'bibtex-modal';
    modal.className = 'bibtex-modal';
    modal.innerHTML = `
      <div class="bibtex-modal-content">
        <span class="bibtex-close">&times;</span>
        <h3>Citation</h3>
        <pre id="bibtex-content"></pre>
        <button id="bibtex-copy" class="bibtex-copy-btn">Copy to Clipboard</button>
      </div>
    `;
    document.body.appendChild(modal);

    // Add close functionality
    const closeBtn = modal.querySelector('.bibtex-close');
    closeBtn.addEventListener('click', function() {
      hideModal();
    });

    // Close modal when clicking outside content
    window.addEventListener('click', function(event) {
      if (event.target === modal) {
        hideModal();
      }
    });

    // Add keyboard escape to close
    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape' && modal.style.display === 'block') {
        hideModal();
      }
    });

    // Helper function to properly hide modal
    function hideModal() {
      modal.style.display = 'none';
      // Remove any event listeners added during display
      const oldModal = document.getElementById('bibtex-modal');
      if (oldModal) {
        const newModal = oldModal.cloneNode(true);
        oldModal.parentNode.replaceChild(newModal, oldModal);
        // Re-add close functionality
        const newCloseBtn = newModal.querySelector('.bibtex-close');
        if (newCloseBtn) {
          newCloseBtn.addEventListener('click', hideModal);
        }
        // Re-add copy functionality
        const newCopyBtn = document.getElementById('bibtex-copy');
        if (newCopyBtn) {
          newCopyBtn.addEventListener('click', function() {
            const content = document.getElementById('bibtex-content').textContent;
            navigator.clipboard.writeText(content)
              .then(() => {
                const originalText = newCopyBtn.textContent;
                newCopyBtn.textContent = 'Copied!';
                newCopyBtn.classList.add('copied');
                setTimeout(() => {
                  newCopyBtn.textContent = originalText;
                  newCopyBtn.classList.remove('copied');
                }, 2000);
              })
              .catch(err => {
                console.error('Could not copy text: ', err);
              });
          });
        }
      }
    }

    // Add copy functionality
    const copyBtn = document.getElementById('bibtex-copy');
    copyBtn.addEventListener('click', function() {
      const content = document.getElementById('bibtex-content').textContent;
      navigator.clipboard.writeText(content)
        .then(() => {
          const originalText = copyBtn.textContent;
          copyBtn.textContent = 'Copied!';
          copyBtn.classList.add('copied');
          setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.classList.remove('copied');
          }, 2000);
        })
        .catch(err => {
          console.error('Could not copy text: ', err);
        });
    });
  }

  // Fetch and cache the bibtex data
  let bibtexData = {};

  // Try multiple possible paths for the BibTeX file
  const bibtexPaths = [
    '../bibtex/papers.bib',
    'bibtex/papers.bib',
    '/bibtex/papers.bib'
  ];

  let bibtexLoaded = false;

  function loadBibtex(index) {
    if (index >= bibtexPaths.length) {
      console.error('Could not load BibTeX data from any path');
      // Set up the buttons anyway but with error handling
      setupButtons();
      return;
    }

    const path = bibtexPaths[index];
    console.log(`Trying to load BibTeX from: ${path}`);

    fetch(path)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.text();
      })
      .then(data => {
        console.log(`Successfully loaded BibTeX data from: ${path}`);
        bibtexLoaded = true;

        // Simple parser for bibtex entries
        const entries = data.split('@');
        for (let entry of entries) {
          if (entry.trim() === '') continue;

          // Extract the citation key
          const match = entry.match(/^\w+{([^,]+),/);
          if (match && match[1]) {
            const key = match[1];
            bibtexData[key] = '@' + entry;
            console.log(`Loaded citation for key: ${key}`);
          }
        }

        // Set up the buttons with the loaded data
        setupButtons();
      })
      .catch(error => {
        console.error(`Error loading BibTeX from ${path}:`, error);
        // Try the next path
        loadBibtex(index + 1);
      });
  }

  function setupButtons() {
    // Add click event to all buttons
    bibtexButtons.forEach(button => {
      button.addEventListener('click', function(e) {
        e.preventDefault();
        const bibtexId = this.getAttribute('data-bibtex-id');
        const bibtexContent = document.getElementById('bibtex-content');

        // Make sure modal is reset before showing
        modal.style.display = 'none';
        setTimeout(() => {
          if (bibtexData[bibtexId]) {
            bibtexContent.textContent = bibtexData[bibtexId];
          } else {
            if (bibtexLoaded) {
              bibtexContent.textContent = `Citation key "${bibtexId}" not found in the BibTeX file`;
            } else {
              bibtexContent.textContent = `BibTeX file could not be loaded. Citation key: ${bibtexId}`;
            }
          }
          modal.style.display = 'block';
        }, 10);
      });
    });
  }

  // Start loading BibTeX data
  loadBibtex(0);
});
