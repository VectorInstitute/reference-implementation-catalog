document.addEventListener('DOMContentLoaded', function() {
  // Initialize tooltips for tags
  tippy('[data-tippy]', {
    content: function(reference) {
      return reference.getAttribute('data-tippy');
    },
    placement: 'top',
    arrow: true,
    animation: 'scale'
  });

  // Make cards fully clickable
  const cards = document.querySelectorAll('.grid.catalog-cards .card');
  cards.forEach(card => {
    const link = card.querySelector('h3 a');
    if (link) {
      card.style.cursor = 'pointer';
      card.addEventListener('click', function(e) {
        // Only navigate if the click wasn't on another link or interactive element
        if (!e.target.closest('a:not(h3 a)') &&
            !e.target.closest('button') &&
            !e.target.closest('input')) {
          window.location.href = link.getAttribute('href');
        }
      });
    }
  });
});
