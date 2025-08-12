document.addEventListener('DOMContentLoaded', () => {
    // Example: Smooth scroll for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Example: Sticky header class on scroll
    const header = document.querySelector('.main-header');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) { // Add class after scrolling 50px
            header.classList.add('sticky');
        } else {
            header.classList.remove('sticky');
        }
    });

    // You might add a class in CSS like:
    /*
    .main-header.sticky {
        background-color: rgba(33, 37, 41, 0.95);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    */
});