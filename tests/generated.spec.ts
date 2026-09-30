import { test, expect } from '@playwright/test';

test.describe('Login Tests', () => {
    test('Verify Successful Login Redirects to Homepage', async ({ page }) => {
        // Step 1: Open the application in a web browser
        await page.goto('https://practicetestautomation.com/practice-test-login/');

        // Step 2: Enter the valid username into the "Username" input field
        await page.fill('input[name="username"]', 'student');

        // Step 3: Enter the valid password into the "Password" input field
        await page.fill('input[name="password"]', 'Password123');

        // Step 4: Click the "Login" button
        await page.click('text=Login');

        // Expected Result: Wait for the homepage to load
        await expect(page).toHaveURL('https://practicetestautomation.com/practice-test-login/');

        // Check if homepage loaded successfully
        await expect(page.locator('h1')).toHaveText('Welcome'); // Adjust this line based on actual welcome message
    });
});