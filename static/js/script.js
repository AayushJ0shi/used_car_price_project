// Form validation and enhancements
document.addEventListener('DOMContentLoaded', function() {
    // Brand-Model dependency
    const brandSelect = document.getElementById('brand');
    const modelSelect = document.getElementById('car_model');
    
    if (brandSelect && modelSelect) {
        brandSelect.addEventListener('change', function() {
            const brand = this.value;
            // Clear model selection when brand changes
            modelSelect.innerHTML = '<option value="">Select Model</option>';
            
            if (brand) {
                // In a real application, you would fetch models from server
                // For now, we'll rely on the server-side rendering
                console.log('Brand selected:', brand);
            }
        });
    }

    // Add real-time form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const inputs = this.querySelectorAll('input[required], select[required]');
            let valid = true;
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!valid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    });
});