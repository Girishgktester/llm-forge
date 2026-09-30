import { test, expect } from '@playwright/test';

test.describe('Login Functionality Tests', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://practicetestautomation.com/practice-test-login/');
    });

    test('User logs in to the application using valid credentials', async ({ page }) => {
        await page.fill('input[name="username"]', 'student');
        await page.fill('input[name="password"]', 'Password123');
        await page.click('button[type="submit"]');

        await expect(page).toHaveURL(/.*welcome/);
        await expect(page.locator('h1')).toHaveText('Welcome');
    });

    test('User fails to log in with invalid username', async ({ page }) => {
        await page.fill('input[name="username"]', 'invalidUser');
        await page.fill('input[name="password"]', 'Password123');
        await page.click('button[type="submit"]');

        await expect(page.locator('.error')).toHaveText('Invalid username or password.');
    });

    test('User fails to log in with invalid password', async ({ page }) => {
        await page.fill('input[name="username"]', 'student');
        await page.fill('input[name="password"]', 'wrongPassword');
        await page.click('button[type="submit"]');

        await expect(page.locator('.error')).toHaveText('Invalid username or password.');
    });

    test('User fails to log in with both fields empty', async ({ page }) => {
        await page.click('button[type="submit"]');

        await expect(page.locator('.error')).toHaveText('Please enter your username and password.');
    });

    test('User logs in successfully and checks for logout option', async ({ page }) => {
        await page.fill('input[name="username"]', 'student');
        await page.fill('input[name="password"]', 'Password123');
        await page.click('button[type="submit"]');

        await expect(page).toHaveURL(/.*welcome/);
        await expect(page.locator('h1')).toHaveText('Welcome');
        await expect(page.locator('a.logout')).toBeVisible();
    });

    test('User fails to log in with SQL injection attack', async ({ page }) => {
        await page.fill('input[name="username"]', "student' OR '1'='1");
        await page.fill('input[name="password"]', "Password123' OR '1'='1");
        await page.click('button[type="submit"]');

        await expect(page.locator('.error')).toHaveText('Invalid username or password.');
    });
});