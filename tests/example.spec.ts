import { test, expect } from '@playwright/test';

test('homepage has the expected title', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Playwright/);
});

test('get started link opens the installation page', async ({ page }) => {
  await page.goto('/');

  await page.getByRole('link', { name: 'Get started' }).click();

  await expect(
    page.getByRole('heading', { name: 'Installation' })
  ).toBeVisible();
});
